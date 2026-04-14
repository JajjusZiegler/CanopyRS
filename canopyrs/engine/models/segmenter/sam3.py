from typing import List, Optional
import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from transformers import (
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3TrackerVideoInferenceSession,
)
from pathlib import Path

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_box
from canopyrs.engine.multispectral import calculate_vi, vi_to_sam_input, select_best_masks


@SEGMENTER_REGISTRY.register('sam3')
class Sam3PredictorWrapper(SegmenterWrapperBase):
    """
    SAM3 Tracker (Video) wrapper.

    facebook/sam3 is a sam3_video checkpoint that uses Sam3TrackerVideoModel
    and a session-based inference API. Each image is treated as a single-frame
    "video". Box prompts are provided per-object via the processor's
    process_new_points_or_boxes_for_video_frame helper, and the model forward
    pass returns one mask per object.

    Multispectral support (Path A – Early Resampling):
        When ``forward()`` receives a non-None ``ms_images`` list and
        ``self.config.ms_index_type`` is set, it runs an additional inference
        pass on each tile using a vegetation-index-derived 3-channel grayscale
        image.  The mask with the higher predicted IoU score is selected for
        each detected crown and passed to the post-processing queue.
    """

    MODEL_MAPPING = {
        't': "facebook/sam3",
        's': "facebook/sam3",
        'b': "facebook/sam3",
        'l': "facebook/sam3",
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.MODEL_MAPPING[self.config.architecture]

        print(f"Loading SAM3 model {self.model_name}")
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(self.model_name)
        self.model = Sam3TrackerVideoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"SAM3 model {self.model_name} loaded")

        self.score_threshold = getattr(self.config, "sam3_score_threshold", 0.0)
        self.mask_threshold = getattr(self.config, "sam3_mask_threshold", 0.0)
        # Resize tiles to this resolution before inference (matches original code intent)
        self.target_tile_size = getattr(self.config, "target_tile_size", 1777)
        print(f"SAM3 will resize tiles to {self.target_tile_size}x{self.target_tile_size} for processing")

        checkpoint_path = getattr(self.config, 'checkpoint_path', None)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------ #
    #  Checkpoint helpers (unchanged from original)                       #
    # ------------------------------------------------------------------ #

    def _load_checkpoint(self, checkpoint_path):
        checkpoint_path_str = str(checkpoint_path)

        if checkpoint_path_str.startswith("https://huggingface.co/") or \
                checkpoint_path_str.startswith("http://huggingface.co/"):
            local_path = self._download_from_huggingface(checkpoint_path_str)
            if local_path is None:
                print(f"\n⚠️  WARNING: Failed to download checkpoint from: {checkpoint_path_str}")
                print("   Using base pretrained model instead.\n")
                return
        else:
            local_path = Path(checkpoint_path_str)
            if not local_path.exists():
                print(f"\n⚠️  WARNING: Checkpoint not found: {checkpoint_path_str}")
                print("   Using base pretrained model instead.\n")
                return

        print(f"\n{'='*60}")
        print(f"Loading fine-tuned checkpoint:")
        print(f"  Path: {local_path}")

        state_dict = torch.load(local_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            model_state_dict = state_dict['model_state_dict']
            print("  Checkpoint type: Full training checkpoint")
            if 'epoch' in state_dict:
                print(f"  Epoch: {state_dict['epoch']}")
        else:
            model_state_dict = state_dict
            print("  Checkpoint type: Model weights only")

        self.model.load_state_dict(model_state_dict, strict=False)
        print("✓ Fine-tuned weights loaded successfully!")
        print(f"{'='*60}\n")

    def _download_from_huggingface(self, url: str) -> "Path | None":
        from huggingface_hub import hf_hub_download
        import re

        pattern = r"https?://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)"
        match = re.match(pattern, url)
        if not match:
            print(f"  Could not parse HuggingFace URL: {url}")
            return None

        repo_id, revision, filename = match.group(1), match.group(2), match.group(3)
        print(f"\n{'='*60}")
        print(f"Downloading checkpoint from HuggingFace:")
        print(f"  Repo: {repo_id}  Revision: {revision}  File: {filename}")
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
            print(f"  Downloaded to: {local_path}")
            return Path(local_path)
        except Exception as e:
            print(f"  Download failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Main inference loop                                                #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        images: List[np.ndarray],
        boxes: List[np.ndarray],
        boxes_object_ids: List[int],
        tiles_idx: List[int],
        queue: multiprocessing.JoinableQueue,
        ms_images: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """
        images: list of CxHxW np arrays (C at least 3)
        boxes:  list of (Ni, 4) xyxy boxes per image
        boxes_object_ids: list of object-id lists aligned with boxes
        tiles_idx: list of tile indices
        ms_images: optional list of CxHxW float32 MS tile arrays (one per image,
                   or None entries).  When provided and config.ms_index_type is
                   set, dual-stream inference is performed and the best mask per
                   box is selected by predicted IoU score.
        """
        ms_index_type = getattr(self.config, 'ms_index_type', None)
        use_ms = ms_images is not None and ms_index_type is not None

        ms_iter = ms_images if use_ms else [None] * len(images)

        for image, image_boxes, image_boxes_object_ids, tile_idx, ms_image in zip(
            images, boxes, boxes_object_ids, tiles_idx, ms_iter
        ):
            # C,H,W → H,W,C, RGB only
            image = image[:3, :, :].transpose((1, 2, 0))
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            orig_H, orig_W, _ = image.shape

            # Optionally resize to target_tile_size
            if orig_H != self.target_tile_size or orig_W != self.target_tile_size:
                import cv2
                image = cv2.resize(
                    image, (self.target_tile_size, self.target_tile_size),
                    interpolation=cv2.INTER_LINEAR
                )
                scale_x = self.target_tile_size / orig_W
                scale_y = self.target_tile_size / orig_H
                image_boxes = image_boxes.astype(np.float32)
                image_boxes[:, [0, 2]] *= scale_x
                image_boxes[:, [1, 3]] *= scale_y

            H, W, _ = image.shape

            if len(image_boxes) == 0:
                continue

            # Prepare MS SAM input if applicable
            ms_pil_image = None
            if use_ms and ms_image is not None:
                ms_pil_image = self._prepare_ms_pil_image(
                    ms_image, target_H=H, target_W=W
                )

            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")
            ):
                pil_image = Image.fromarray(image).convert("RGB")
                n_masks_processed = 0

                for i in range(0, len(image_boxes), self.config.box_batch_size):
                    box_batch = np.asarray(
                        image_boxes[i: i + self.config.box_batch_size], dtype=np.float32
                    )
                    box_object_ids_batch = image_boxes_object_ids[
                        i: i + self.config.box_batch_size
                    ]

                    # Clip and filter degenerate boxes
                    box_batch[:, 0] = np.clip(box_batch[:, 0], 0, W)
                    box_batch[:, 1] = np.clip(box_batch[:, 1], 0, H)
                    box_batch[:, 2] = np.clip(box_batch[:, 2], 0, W)
                    box_batch[:, 3] = np.clip(box_batch[:, 3], 0, H)
                    valid = (box_batch[:, 2] > box_batch[:, 0]) & (
                        box_batch[:, 3] > box_batch[:, 1]
                    )
                    if not valid.any():
                        continue
                    box_batch = box_batch[valid]
                    box_object_ids_batch = [
                        oid for oid, v in zip(box_object_ids_batch, valid) if v
                    ]

                    masks, scores = self._ensemble_predict(self._predict_batch, pil_image, box_batch)

                    if masks is None or len(masks) == 0:
                        continue

                    # MS inference (optional)
                    if ms_pil_image is not None:
                        ms_masks, ms_scores = self._ensemble_predict(self._predict_batch, ms_pil_image, box_batch)
                        if ms_masks is not None and len(ms_masks) == len(masks):
                            masks, scores = select_best_masks(
                                masks, scores, ms_masks, ms_scores
                            )

                    # Scale masks back to original resolution when tile was resized
                    if orig_H != self.target_tile_size or orig_W != self.target_tile_size:
                        import cv2
                        masks = np.array([
                            cv2.resize(m, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
                            for m in masks
                        ])

                    output_size = (orig_H, orig_W)
                    n_masks_processed = self.queue_masks(
                        box_object_ids_batch,
                        masks,
                        output_size,
                        scores,
                        tile_idx,
                        n_masks_processed,
                        queue,
                    )

    # ------------------------------------------------------------------ #
    #  Single-batch prediction                                            #
    # ------------------------------------------------------------------ #

    def _predict_batch(
        self, image: Image.Image, boxes: np.ndarray, image_size: tuple = None
    ):
        """
        Run SAM3TrackerVideo on a single image with Nq box prompts.

        The image is treated as a 1-frame video. Each box defines one
        tracked object. Returns one mask per box.

        Args:
            image:      PIL.Image (H x W x 3)
            boxes:      (Nq, 4) float32 [x1, y1, x2, y2]
            image_size: (H, W) of the (possibly resized) image

        Returns:
            masks:  (Nq, H, W) uint8
            scores: (Nq,)   float32
        """
        if len(boxes) == 0:
            return None, None

        if image_size is None:
            W_pil, H_pil = image.size  # PIL uses (W, H)
            H, W = H_pil, W_pil
        else:
            H, W = image_size
        n_boxes = len(boxes)
        obj_ids = list(range(n_boxes))

        # Encode the image into pixel_values (shape: 1 x C x H_enc x W_enc)
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        # dtype must be a torch.dtype (not a string) for .to() inside add_new_frame
        session_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # Build a 1-frame inference session
        session = Sam3TrackerVideoInferenceSession(
            video=None,
            video_height=H,
            video_width=W,
            inference_device=self.device,
            inference_state_device=self.device,
            video_storage_device=self.device,
            dtype=session_dtype,
        )

        # Add the single frame
        frame_idx = session.add_new_frame(pixel_values, frame_idx=0)

        # Register box prompts: input_boxes = [[box0, box1, ...]] shape [1, Nq, 4]
        input_boxes = [[box.tolist() for box in boxes]]  # [1, Nq, 4]
        self.processor.process_new_points_or_boxes_for_video_frame(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=obj_ids,
            input_boxes=input_boxes,
            original_size=(H, W),
        )

        # Run model — forward returns masks already at (video_height, video_width)
        output = self.model(session, frame_idx=frame_idx)

        # pred_masks: (Nq, 1, H, W) logits at session resolution
        pred_masks = output.pred_masks  # kept on device

        # Binarise with threshold
        masks_binary = (pred_masks.float() > self.mask_threshold)  # (Nq, 1, H, W)
        if masks_binary.ndim == 4 and masks_binary.shape[1] == 1:
            masks_binary = masks_binary[:, 0]  # (Nq, H, W)
        masks = masks_binary.to(torch.uint8).cpu().numpy()

        # Object scores: sigmoid of logits → [0, 1]
        scores = torch.sigmoid(
            output.object_score_logits.float()
        ).detach().cpu().numpy()
        if scores.shape[0] != n_boxes:
            scores = np.ones(n_boxes, dtype=np.float32)

        return masks, scores

    def _prepare_ms_pil_image(
        self,
        ms_data: np.ndarray,
        target_H: int,
        target_W: int,
    ) -> Image.Image:
        """
        Convert a multi-band MS tile to a 3-channel grayscale PIL Image for SAM.

        The method computes the configured vegetation index, converts it to
        8-bit, and stacks three identical copies so that SAM's processor can
        accept it as a standard RGB image.

        Args:
            ms_data: CxHxW float32 array (all bands of the MS tile).
            target_H: Target height (should match the RGB tile after resizing).
            target_W: Target width.

        Returns:
            PIL.Image of size (target_W, target_H) in RGB mode.
        """
        import cv2

        vi = calculate_vi(
            ms_data,
            index_type=self.config.ms_index_type,
            nir_band_idx=getattr(self.config, 'ms_nir_band_idx', 4),
            red_band_idx=getattr(self.config, 'ms_red_band_idx', 2),
            green_band_idx=getattr(self.config, 'ms_green_band_idx', 1),
            blue_band_idx=getattr(self.config, 'ms_blue_band_idx', 0),
            red_edge_band_idx=getattr(self.config, 'ms_red_edge_band_idx', None),
        )

        # Convert VI to 3-channel uint8 SAM input (H, W, 3)
        ms_sam = vi_to_sam_input(vi)

        # Resize to match the (possibly upscaled) RGB tile dimensions
        if ms_sam.shape[0] != target_H or ms_sam.shape[1] != target_W:
            ms_sam = cv2.resize(ms_sam, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

        return Image.fromarray(ms_sam).convert("RGB")

    def infer_on_dataset(
        self,
        dataset: DetectionLabeledRasterCocoDataset,
        ms_tiles_path: Optional[str] = None,
    ):
        return self._infer_on_dataset(
            dataset,
            collate_fn_infer_image_box,
            ms_tiles_path=ms_tiles_path,
        )
