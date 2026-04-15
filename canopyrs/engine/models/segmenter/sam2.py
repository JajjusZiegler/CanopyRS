from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from sam2.sam2_image_predictor import SAM2ImagePredictor

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_box
from canopyrs.engine.multispectral import calculate_vi, vi_to_sam_input, select_best_masks
from pathlib import Path


@SEGMENTER_REGISTRY.register('sam2')
class Sam2PredictorWrapper(SegmenterWrapperBase):
    """
    SAM2 image predictor wrapper.

    Multispectral support (Path A – Early Resampling):
        When ``forward()`` receives a non-None ``ms_images`` list and
        ``self.config.ms_index_type`` is set, it runs an additional inference
        pass on each tile using a vegetation-index-derived 3-channel grayscale
        image.  The mask with the higher predicted IoU score is selected for
        each detected crown.
    """

    MODEL_MAPPING = {
        't': "facebook/sam2-hiera-tiny",
        's': "facebook/sam2-hiera-small",
        'b': "facebook/sam2-hiera-base-plus",
        'l': "facebook/sam2-hiera-large",
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):

        super().__init__(config)

        self.model_name = self.MODEL_MAPPING[self.config.architecture]

        # Load SAM model and processor
        print(f"Loading model {self.model_name}")
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
        print(f"Model {self.model_name} loaded")

        checkpoint_path = getattr(self.config, 'checkpoint_path', None)
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                print(f"Loading fine-tuned checkpoint:")
                print(f"  Path: {checkpoint_path}")
                
                # Load state dict
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    # Full checkpoint with optimizer, etc.
                    model_state_dict = state_dict['model_state_dict']
                    print(f"  Checkpoint type: Full training checkpoint")
                else:
                    # Just model weights
                    model_state_dict = state_dict
                    print(f"  Checkpoint type: Model weights only")
                
                # Load weights into predictor model
                self.predictor.model.load_state_dict(model_state_dict)
                print(f"Fine-tuned weights loaded successfully!")
            else:
                print(f"\nWARNING: Checkpoint not found: {checkpoint_path}")
                print(f"Using base pretrained model instead.\n")

    def forward(
        self,
        images: List[np.array],
        boxes: List[np.array],
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

        # Only 1 image per batch supported for now
        for image, image_boxes, image_boxes_object_ids, tile_idx, ms_image in zip(
            images, boxes, boxes_object_ids, tiles_idx, ms_iter
        ):
            image = image[:3, :, :]
            image = image.transpose((1, 2, 0))
            image = (image * 255).astype(np.uint8)

            H, W = image.shape[:2]

            # Prepare MS SAM input if applicable
            ms_pil_image = None
            if use_ms and ms_image is not None:
                ms_pil_image = self._prepare_ms_pil_image(ms_image, target_H=H, target_W=W)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pil_image = Image.fromarray(image).convert("RGB")
                self.predictor.set_image(pil_image)

                # Process bounding boxes in batches
                n_masks_processed = 0
                for i in range(0, len(image_boxes), self.config.box_batch_size):
                    box_batch = image_boxes[i:i + self.config.box_batch_size]
                    box_object_ids_batch = image_boxes_object_ids[i:i + self.config.box_batch_size]

                    # Define predict_fn for the ensemble; the image embedding
                    # is already set via set_image() above, so the image
                    # argument is ignored inside the closure.
                    def _predict_fn_rgb(image, boxes):
                        m, s, _ = self.predictor.predict(
                            box=boxes, multimask_output=False, normalize_coords=True
                        )
                        if m.ndim == 4:
                            m = m[:, 0, :, :]
                        s_flat = s[:, 0] if s.ndim == 2 else s
                        return m, s_flat

                    rgb_masks, rgb_scores, _ = self._ensemble_predict(_predict_fn_rgb, pil_image, box_batch)

                    # MS inference (optional)
                    if ms_pil_image is not None:
                        self.predictor.set_image(ms_pil_image)

                        def _predict_fn_ms(image, boxes):
                            m, s, _ = self.predictor.predict(
                                box=boxes, multimask_output=False, normalize_coords=True
                            )
                            if m.ndim == 4:
                                m = m[:, 0, :, :]
                            s_flat = s[:, 0] if s.ndim == 2 else s
                            return m, s_flat

                        ms_masks, ms_scores, _ = self._ensemble_predict(_predict_fn_ms, ms_pil_image, box_batch)

                        if ms_masks is not None and len(ms_masks) == len(rgb_masks):
                            rgb_masks, rgb_scores = select_best_masks(
                                rgb_masks, rgb_scores, ms_masks, ms_scores
                            )

                        # Restore the predictor to the RGB image for the next batch
                        self.predictor.set_image(pil_image)

                    image_size = (image.shape[0], image.shape[1])
                    n_masks_processed = self.queue_masks(
                        box_object_ids_batch, rgb_masks, image_size, rgb_scores,
                        tile_idx, n_masks_processed, queue
                    )

    def _prepare_ms_pil_image(
        self,
        ms_data: np.ndarray,
        target_H: int,
        target_W: int,
    ) -> Image.Image:
        """Convert a multi-band MS tile to a 3-channel grayscale PIL Image for SAM."""
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

        ms_sam = vi_to_sam_input(vi)

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
