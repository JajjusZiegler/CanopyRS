"""
U-Net based tree crown delineation segmenter.

Based on the paper:
    Freudenberg, Magdon & Nölke (2022).
    "Individual tree crown delineation in high-resolution remote sensing
     images based on U-Net." Neural Computing and Applications.
    https://doi.org/10.1007/s00521-022-07640-4

The model is a multi-task U-Net that simultaneously predicts:
  1. Tree cover mask
  2. Crown outlines
  3. Distance transform

Individual crowns are extracted via a watershed algorithm that combines
all three outputs.

Model weights (TorchScript .pt files) are available at:
    https://owncloud.gwdg.de/index.php/s/9cUza134XSOwZsB

This wrapper requires:
    pip install scikit-image scipy
    # optional: pip install git+https://github.com/AWF-GAUG/TreeCrownDelineation.git
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

from geodataset.dataset import UnlabeledRasterDataset

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_trivial


# ---------------------------------------------------------------------------
# Helpers – polygon extraction (inline so the package is not required at
# import time; only torch, scipy, scikit-image are needed at runtime)
# ---------------------------------------------------------------------------

def _find_treecrowns(mask: np.ndarray, outlines: np.ndarray, dist: np.ndarray,
                     mask_exp: float = 2, outline_multiplier: float = 5,
                     outline_exp: float = 1, dist_exp: float = 0.5,
                     sigma: float = 1, min_dist: int = 10,
                     binary_threshold: float = 0.1,
                     label_threshold: float = 0.1) -> np.ndarray:
    """
    Combine U-Net outputs into a watershed-labelled array.

    Returns an int32 array where each individual tree crown has a unique
    positive label (background = 0).
    """
    from scipy import ndimage as ndi
    from skimage import filters
    from skimage.feature import corner_peaks
    from skimage.segmentation import watershed

    combined = (np.clip(mask ** mask_exp
                        - outline_multiplier * outlines ** outline_exp, 0, 1)
                * np.clip(dist, 0, 1) ** dist_exp)
    combined = filters.gaussian(combined, sigma=sigma)

    binary_img = combined > binary_threshold

    # local maxima → seed points
    seeds = corner_peaks(
        combined,
        indices=False,
        min_distance=int(min_dist),
        threshold_abs=label_threshold,
        exclude_border=False,
        p_norm=2,
    ).astype(np.int32)
    ndi.label(seeds, output=seeds)

    labels = watershed(-combined, seeds, mask=binary_img, connectivity=2)
    return labels


def _extract_polygons_from_labels(labels: np.ndarray,
                                   area_min: int = 3,
                                   area_max: int = 10 ** 6,
                                   simplify: float = 0.3) -> List[Polygon]:
    """Convert a label image (one label per tree) to a list of Shapely polygons."""
    from rasterio.features import shapes
    from rasterio.transform import IDENTITY

    shape_gen = shapes(labels, labels.astype(bool), transform=IDENTITY)
    polygons = []
    for geom, _ in shape_gen:
        poly = Polygon(geom["coordinates"][0])
        if area_min < poly.area < area_max:
            if simplify:
                poly = poly.simplify(simplify, preserve_topology=True)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
    return polygons


# ---------------------------------------------------------------------------
# Sliding-window inference helper
# ---------------------------------------------------------------------------

def _predict_sliding_window(model: torch.nn.Module,
                             image: np.ndarray,
                             window_size: int,
                             stride: int,
                             device: torch.device,
                             batch_size: int = 8) -> np.ndarray:
    """
    Apply model to a CHW float32 image via overlapping sliding windows.

    Returns a (3, H, W) float32 array (averaged predictions).
    If the image is smaller than window_size in any dimension it is padded
    with zeros (reflect padding) and the result is cropped back.
    """
    c, h, w = image.shape

    # Pad if the tile is smaller than the model's receptive window
    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        c, h, w = image.shape

    out_sum = np.zeros((3, h, w), dtype=np.float32)
    out_cnt = np.zeros((3, h, w), dtype=np.float32)

    # Collect all patch positions
    ys = list(range(0, max(1, h - window_size + 1), stride))
    xs = list(range(0, max(1, w - window_size + 1), stride))

    # Make sure we always cover the bottom-right corner
    if ys[-1] + window_size < h:
        ys.append(h - window_size)
    if xs[-1] + window_size < w:
        xs.append(w - window_size)

    patches = []
    coords = []
    for y in ys:
        y = min(y, h - window_size)
        for x in xs:
            x = min(x, w - window_size)
            patch = image[:, y:y + window_size, x:x + window_size]
            patches.append(patch)
            coords.append((y, x))

    # Batch predict
    for i in range(0, len(patches), batch_size):
        batch_np = np.stack(patches[i:i + batch_size], axis=0)
        batch_t = torch.from_numpy(batch_np).to(dtype=torch.float32, device=device)

        with torch.no_grad():
            pred = model(batch_t)          # (B, 3, ws, ws)

        pred_np = pred.cpu().numpy().astype(np.float32)

        for k, (y, x) in enumerate(coords[i:i + batch_size]):
            out_sum[:, y:y + window_size, x:x + window_size] += pred_np[k]
            out_cnt[:, y:y + window_size, x:x + window_size] += 1.0

    # Average
    out_cnt = np.maximum(out_cnt, 1e-6)
    result = out_sum / out_cnt

    # Crop back to original dimensions if we padded
    orig_h = h - pad_h
    orig_w = w - pad_w
    return result[:, :orig_h, :orig_w]


# ---------------------------------------------------------------------------
# CanopyRS segmenter wrapper
# ---------------------------------------------------------------------------

@SEGMENTER_REGISTRY.register('unet_treecrown')
class UNetTreeCrownWrapper(SegmenterWrapperBase):
    """
    Wrapper for the U-Net tree crown delineation model
    (Freudenberg et al., 2022).

    The model is PROMPT-FREE: it runs on full image tiles and
    returns individual crown polygons via watershed post-processing.

    Configuration keys (all optional unless noted):
        checkpoint_path (str, required): Path(s) to one or more TorchScript
            .pt checkpoint files, separated by '|'.  Multiple models are
            averaged, improving robustness.
        window_size (int): Sliding-window size in pixels.  Default: 256.
        stride (int): Sliding-window stride.  Default: None → window_size - 32.
        batch_size_windows (int): How many window patches to process per GPU
            batch.  Default: 8.
        divide_by (float): Input images are divided by this value before
            forwarding.  Default: 255.
        append_ndvi (bool): Whether to append a computed NDVI channel to the
            input (requires a NIR band).  Default: False.
        red_band (int): 0-based index of the red band.  Default: 0.
        nir_band (int): 0-based index of the NIR band.  Default: 3.
        rescale_ndvi (bool): Rescale NDVI to [0, 1].  Default: False.
        apply_sigmoid (bool): Apply sigmoid to mask and outline outputs
            (needed for models that do NOT include sigmoid internally).
            Default: False.
        min_dist (int): Minimum distance between tree seed points in px.
            Default: 10.
        sigma (float): Gaussian blur σ for watershed map.  Default: 1.
        label_threshold (float): Minimum local-maximum height.  Default: 0.1.
        binary_threshold (float): Background vs. foreground threshold.
            Default: 0.1.
        area_min (int): Minimum crown area in pixels.  Default: 3.
        area_max (int): Maximum crown area in pixels.  Default: 1 000 000.
        simplify_tolerance (float): Polygon simplification distance.
            Default: 0.3.
    """

    REQUIRES_BOX_PROMPT = False

    # ------------------------------------------------------------------ #
    #  Init                                                                #
    # ------------------------------------------------------------------ #

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

        # ---- checkpoint ------------------------------------------------
        checkpoint_path: Optional[str] = getattr(config, 'checkpoint_path', None)
        if not checkpoint_path:
            raise ValueError(
                "UNetTreeCrownWrapper requires 'checkpoint_path' to be set "
                "in the segmenter config (one or more .pt TorchScript files, "
                "separated by '|').\n"
                "Download weights from: "
                "https://owncloud.gwdg.de/index.php/s/9cUza134XSOwZsB"
            )

        model_paths = [p.strip() for p in str(checkpoint_path).split('|')]
        models = []
        for mp in model_paths:
            mp_path = Path(mp)
            if not mp_path.exists():
                raise FileNotFoundError(
                    f"UNetTreeCrown checkpoint not found: {mp_path}"
                )
            print(f"Loading U-Net checkpoint: {mp_path}")
            models.append(torch.jit.load(str(mp_path), map_location=self.device))

        if len(models) == 1:
            self.model = models[0]
        else:
            # Simple averaging wrapper
            self.model = _AveragingModel(models)

        self.model.to(self.device)
        self.model.eval()
        print(f"UNetTreeCrown: loaded {len(models)} model(s).")

        # ---- inference settings ----------------------------------------
        self.window_size: int = int(getattr(config, 'window_size', 256))
        self.stride: int = int(getattr(config, 'stride', self.window_size - 32))
        self.batch_size_windows: int = int(getattr(config, 'batch_size_windows', 8))
        self.divide_by: float = float(getattr(config, 'divide_by', 255.0))
        self.append_ndvi: bool = bool(getattr(config, 'append_ndvi', False))
        self.red_band: int = int(getattr(config, 'red_band', 0))
        self.nir_band: int = int(getattr(config, 'nir_band', 3))
        self.rescale_ndvi: bool = bool(getattr(config, 'rescale_ndvi', False))
        self.apply_sigmoid: bool = bool(getattr(config, 'apply_sigmoid', False))

        # ---- post-processing settings ----------------------------------
        self.min_dist: int = int(getattr(config, 'min_dist', 10))
        self.sigma: float = float(getattr(config, 'sigma', 1))
        self.label_threshold: float = float(getattr(config, 'label_threshold', 0.1))
        self.binary_threshold: float = float(getattr(config, 'binary_threshold', 0.1))
        self.area_min: int = int(getattr(config, 'area_min', 3))
        self.area_max: int = int(getattr(config, 'area_max', 10 ** 6))
        self.simplify_tolerance: float = float(
            getattr(config, 'simplify_tolerance',
                    getattr(config, 'pp_simplify_tolerance', 0.3))
        )

    # ------------------------------------------------------------------ #
    #  Core inference                                                      #
    # ------------------------------------------------------------------ #

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert a CxHxW uint8/float tile to a normalised float32 array."""
        image = image.astype(np.float32)

        if self.divide_by != 1.0:
            image = image / self.divide_by

        if self.append_ndvi:
            red = image[self.red_band]
            nir_ = image[self.nir_band]
            denom = nir_ + red
            ndvi_ch = np.where(denom > 0, (nir_ - red) / denom, 0.0).astype(np.float32)
            if self.rescale_ndvi:
                ndvi_ch = (ndvi_ch + 1.0) / 2.0
            image = np.concatenate([image, ndvi_ch[None]], axis=0)

        # Use only the 3 RGB bands (+ NDVI if appended) — the model expects
        # a fixed number of channels (typically 3 or 4).
        n_expected = 4 if self.append_ndvi else 3
        image = image[:n_expected]

        return image

    def _postprocess_to_polygons(self, prediction: np.ndarray) -> List[Polygon]:
        """
        Run watershed on the 3-channel model output to produce crown polygons.

        Args:
            prediction: (3, H, W) float32 array – (cover, outlines, dist_transform)

        Returns:
            List of Shapely Polygon objects in pixel coordinates.
        """
        if self.apply_sigmoid:
            from scipy.special import expit
            prediction = prediction.copy()
            prediction[0] = expit(prediction[0])   # cover
            prediction[1] = expit(prediction[1])   # outlines
            # dist_transform (channel 2) is left as-is

        mask = prediction[0]
        outlines = prediction[1]
        dist = prediction[2]

        labels = _find_treecrowns(
            mask, outlines, dist,
            sigma=self.sigma,
            min_dist=self.min_dist,
            label_threshold=self.label_threshold,
            binary_threshold=self.binary_threshold,
        )
        polygons = _extract_polygons_from_labels(
            labels,
            area_min=self.area_min,
            area_max=self.area_max,
            simplify=self.simplify_tolerance,
        )
        return polygons

    # ------------------------------------------------------------------ #
    #  SegmenterWrapperBase interface                                      #
    # ------------------------------------------------------------------ #

    def forward(self,
                images: List[np.ndarray],
                boxes=None,
                boxes_object_ids=None,
                tiles_idx: List[int] = None,
                queue: multiprocessing.JoinableQueue = None):
        """
        Not used directly — infer_on_dataset overrides the default loop
        to return polygons without the mask-queue mechanism.
        """
        raise NotImplementedError(
            "UNetTreeCrownWrapper.forward() should not be called directly; "
            "use infer_on_dataset() instead."
        )

    def infer_on_dataset(self, dataset: UnlabeledRasterDataset):
        """
        Run the U-Net on all tiles in the dataset.

        Returns the same tuple as SegmenterWrapperBase._infer_on_dataset:
            (tiles_paths, tiles_masks_objects_ids,
             tiles_masks_polygons, tiles_masks_scores)
        """
        infer_dl = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn_trivial,
            num_workers=0,   # avoid multiprocessing issues with TorchScript
        )

        tiles_paths = []
        tiles_masks_polygons = []
        tiles_masks_scores = []

        for tile_idx, sample in enumerate(
            tqdm(infer_dl, desc="UNet TreeCrown inference…", leave=True)
        ):
            # sample is a list of one CxHxW tensor (from collate_fn_trivial)
            image_tensor = sample[0]  # (C, H, W) float tensor in [0,1]
            if isinstance(image_tensor, torch.Tensor):
                image_np = image_tensor.numpy()
            else:
                image_np = np.array(image_tensor)

            tile_path = dataset.tile_paths[tile_idx]
            tiles_paths.append(tile_path)

            # ---- preprocess -----------------------------------------
            image_np = self._preprocess(image_np)  # (C, H, W) float32

            # ---- sliding-window inference ---------------------------
            stride = self.stride or (self.window_size - 32)
            prediction = _predict_sliding_window(
                model=self.model,
                image=image_np,
                window_size=self.window_size,
                stride=stride,
                device=self.device,
                batch_size=self.batch_size_windows,
            )  # (3, H, W)

            # ---- watershed post-processing --------------------------
            polygons = self._postprocess_to_polygons(prediction)

            tiles_masks_polygons.append(polygons)
            tiles_masks_scores.append([1.0] * len(polygons))

            print(f"  Tile {tile_idx + 1}: {len(polygons)} crowns found.")

        # Return None for object ids (no box prompts) — SegmenterComponent will
        # auto-assign unique IDs, consistent with _infer_on_dataset convention.
        return tiles_paths, None, tiles_masks_polygons, tiles_masks_scores


# ---------------------------------------------------------------------------
# Simple model ensemble (averages outputs of multiple TorchScript models)
# ---------------------------------------------------------------------------

class _AveragingModel(torch.nn.Module):
    """Averages the float outputs of N TorchScript models."""

    def __init__(self, models: List[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)
