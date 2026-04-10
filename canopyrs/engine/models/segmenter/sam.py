from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from transformers import SamModel, SamProcessor

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_box
from canopyrs.engine.multispectral import calculate_vi, vi_to_sam_input, select_best_masks


@SEGMENTER_REGISTRY.register('sam')
class SamPredictorWrapper(SegmenterWrapperBase):
    """
    SAM v1 wrapper.

    Multispectral support (Path A – Early Resampling):
        When ``forward()`` receives a non-None ``ms_images`` list and
        ``self.config.ms_index_type`` is set, it runs an additional inference
        pass on each tile using a vegetation-index-derived 3-channel grayscale
        image.  The mask with the higher predicted IoU score is selected for
        each detected crown.
    """

    MODEL_TYPE_MAPPING = {
        'b': "facebook/sam-vit-base",
        'l': "facebook/sam-vit-large",
        'h': "facebook/sam-vit-huge"
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

        self.model_name = self.MODEL_TYPE_MAPPING[self.config.architecture]

        # Load SAM model and processor
        print(f"Loading model {self.model_name}")
        self.model = SamModel.from_pretrained(self.model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(self.model_name)
        print(f"Model {self.model_name} loaded")

    def forward(
        self,
        images: List[np.array],
        boxes: List[np.array],
        boxes_object_ids: List[np.array],
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
                   set, dual-stream inference is performed.
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
            pil_image = Image.fromarray(image).convert("RGB")
            image_boxes_list = image_boxes.tolist()

            # Prepare MS SAM input if applicable
            ms_pil_image = None
            if use_ms and ms_image is not None:
                ms_pil_image = self._prepare_ms_pil_image(ms_image, target_H=H, target_W=W)

            # Process bounding boxes in batches
            n_masks_processed = 0
            for i in range(0, len(image_boxes_list), self.config.box_batch_size):
                box_batch = image_boxes_list[i:i + self.config.box_batch_size]
                boxes_object_ids_batch = image_boxes_object_ids[i:i + self.config.box_batch_size]

                inputs = self.processor(pil_image, input_boxes=[box_batch], return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, multimask_output=False)

                # Post-process RGB masks
                rgb_masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )
                rgb_masks = rgb_masks[0]
                if rgb_masks.ndim == 4:
                    rgb_masks = rgb_masks[:, 0, :, :]
                rgb_masks = rgb_masks.cpu().numpy().astype(np.uint8)
                rgb_scores = outputs.iou_scores.cpu()[0]

                # MS inference (optional)
                if ms_pil_image is not None:
                    ms_inputs = self.processor(ms_pil_image, input_boxes=[box_batch], return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        ms_outputs = self.model(**ms_inputs, multimask_output=False)

                    ms_masks = self.processor.image_processor.post_process_masks(
                        ms_outputs.pred_masks,
                        ms_inputs["original_sizes"],
                        ms_inputs["reshaped_input_sizes"]
                    )
                    ms_masks = ms_masks[0]
                    if ms_masks.ndim == 4:
                        ms_masks = ms_masks[:, 0, :, :]
                    ms_masks = ms_masks.cpu().numpy().astype(np.uint8)
                    ms_scores = ms_outputs.iou_scores.cpu()[0]

                    rgb_masks, rgb_scores = select_best_masks(
                        rgb_masks,
                        rgb_scores.numpy(),
                        ms_masks,
                        ms_scores.numpy(),
                    )
                    rgb_scores = torch.from_numpy(rgb_scores)

                image_size = (image.shape[0], image.shape[1])

                n_masks_processed = self.queue_masks(
                    boxes_object_ids_batch, rgb_masks, image_size, rgb_scores,
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
