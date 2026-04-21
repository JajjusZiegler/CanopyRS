from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Optional
import multiprocessing
import warnings
import cv2
import numpy as np
import psutil
import rasterio
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"None of the inputs have requires_grad=True\. Gradients will be None"
)
from shapely import box
from shapely.affinity import scale
from tqdm import tqdm

from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset, BaseDataset
from geodataset.utils import mask_to_polygon

from canopyrs.engine.config_parsers import SegmenterConfig


def get_memory_usage():
    memory_info = psutil.virtual_memory()
    memory_percentage = memory_info.percent

    return memory_percentage


def process_masks(queue,
                  output_dict,
                  output_dict_lock,
                  simplify_tolerance,
                  remove_rings,
                  remove_small_geoms,
                  processed_counter):
    results = {}
    while True:
        item = queue.get()
        if item is None:
            break
        tile_idx, mask_ids, box_object_ids, masks, scores, image_size = item
        try:
            masks_polygons = [mask_to_polygon(mask,
                                              simplify_tolerance=simplify_tolerance,
                                              remove_rings=remove_rings,
                                              remove_small_geoms=remove_small_geoms) for mask in masks]

            # Fix invalid polygons
            for id, polygon in enumerate(masks_polygons):
                if not polygon.is_valid:
                    # If the polygon is still invalid, set its score to 0 and create a dummy box polygon
                    polygon = box(0, 0, 1, 1)
                    scores[id] = 0.0
                if polygon.is_empty:
                    # If the polygon is empty, set its score to 0 and create a dummy box polygon
                    polygon = box(0, 0, 1, 1)
                    scores[id] = 0.0
                masks_polygons[id] = polygon

            mask_h, mask_w = masks.shape[-2], masks.shape[-1]  # e.g. 28,28
            orig_h, orig_w = image_size[0], image_size[1]  # e.g. 1024,1024
            if (mask_h != orig_h) or (mask_w != orig_w):
                # Compute scaling factors for x (width) and y (height)
                scale_x = float(orig_w) / float(mask_w)
                scale_y = float(orig_h) / float(mask_h)
                resized_polygons = []
                for poly in masks_polygons:
                    # Scale shapely polygon from (0,0)
                    poly_scaled = scale(poly, xfact=scale_x, yfact=scale_y, origin=(0, 0))
                    resized_polygons.append(poly_scaled)

                masks_polygons = resized_polygons

            # Store the tile/image results
            if tile_idx not in results:
                results[tile_idx] = []
            [results[tile_idx].append((mask_id, box_object_id, mask_poly, score)) for mask_id, box_object_id, mask_poly, score in
             zip(mask_ids, box_object_ids, masks_polygons, scores)]

            with processed_counter.get_lock():
                processed_counter.value += 1
        except Exception as e:
            import traceback
            print(f"[process_masks] ERROR on tile_idx={tile_idx}, masks={masks.shape}: {e}\n{traceback.format_exc()}",
                  flush=True)
        finally:
            queue.task_done()  # Always signal completion to unblock queue.join()

    with output_dict_lock:
        for tile_idx in results:
            if tile_idx not in output_dict:
                output_dict[tile_idx] = results[tile_idx]
            else:
                current_list = output_dict[tile_idx]
                current_list.extend(results[tile_idx])
                output_dict[tile_idx] = current_list


class SegmenterWrapperBase(ABC):
    REQUIRES_BOX_PROMPT = None

    def __init__(self, config: SegmenterConfig):
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._check_init()

    def _check_init(self):
        assert self.REQUIRES_BOX_PROMPT is not None,\
            "Classes built from SegmenterWrapperBase must have REQUIRES_BOX_PROMPT set to True or False"

    @abstractmethod
    def forward(
        self,
        images: List[np.array],
        boxes: List[np.array],
        boxes_object_ids: List[int or None],
        tiles_idx: List[int],
        queue: multiprocessing.JoinableQueue,
        ms_images: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """
        Run inference on a batch of image tiles.

        Args:
            images: List of CxHxW numpy arrays (primary RGB tiles).
            boxes: List of (N, 4) xyxy box arrays per tile.
            boxes_object_ids: List of object-ID lists aligned with boxes.
            tiles_idx: List of tile indices (for result assembly).
            queue: JoinableQueue for posting mask results to post-process workers.
            ms_images: Optional list of CxHxW numpy arrays (multispectral tiles,
                       one per image, or ``None`` entries for tiles without an MS
                       counterpart).  When provided and ``self.config.ms_index_type``
                       is set, the wrapper computes a vegetation index, converts it
                       to a 3-channel grayscale SAM input, runs a second inference
                       pass, and selects the best mask per box based on IoU score.
        """
        pass

    @abstractmethod
    def infer_on_dataset(
        self,
        dataset: BaseDataset,
        ms_tiles_path: Optional[str] = None,
    ):
        """
        Run inference over an entire dataset.

        Args:
            dataset: Dataset of tiles (labeled or unlabeled).
            ms_tiles_path: Optional path to the directory containing resampled
                           multispectral tiles.  When provided, the corresponding
                           MS tile for each RGB tile is loaded and passed to
                           ``forward()`` as ``ms_images``.
        """
        pass

    @staticmethod
    def _jitter_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
        """
        Randomly perturb each box corner by ±scale × side_length.

        Args:
            boxes: (N, 4) float32 array of [x1, y1, x2, y2] boxes.
            scale: Fraction of the box side-length used as the noise magnitude.

        Returns:
            Jittered copy of *boxes* (same shape, float32).
        """
        boxes = boxes.copy().astype(np.float32)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        boxes[:, 0] += np.random.uniform(-scale, scale, len(boxes)) * widths
        boxes[:, 1] += np.random.uniform(-scale, scale, len(boxes)) * heights
        boxes[:, 2] += np.random.uniform(-scale, scale, len(boxes)) * widths
        boxes[:, 3] += np.random.uniform(-scale, scale, len(boxes)) * heights
        return boxes

    def _ensemble_predict(
        self,
        predict_fn,
        image,
        boxes: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run *predict_fn* ``ensemble_n_runs`` times (optionally with box jitter)
        and aggregate the results.

        Args:
            predict_fn: Callable ``(image, boxes) -> (masks, scores)`` where
                *masks* is ``(N, H, W)`` uint8 and *scores* is ``(N,)`` float32.
            image: Image argument forwarded to *predict_fn* unchanged on every
                run.  For SAM2 the image embedding is pre-computed by the caller
                via ``set_image()``; *image* is accepted for API uniformity but
                may be ignored inside *predict_fn*.
            boxes: ``(N, 4)`` float32 array of [x1, y1, x2, y2] box prompts.

        Returns:
            ``(masks, scores, prob_map)`` — *prob_map* is ``(N, H, W)`` float32
            with per-pixel crown probabilities averaged across ensemble runs
            (only set when ``ensemble_method == 'heatmap'`` and ``n_runs > 1``,
            otherwise ``None``).  *masks* and *scores* match the dtype/shape
            contract of *predict_fn*, or ``(None, None, None)`` if every run
            returned no masks.
        """
        n_runs = getattr(self.config, 'ensemble_n_runs', 1)
        jitter_scale = getattr(self.config, 'ensemble_box_jitter_scale', 0.0)
        method = getattr(self.config, 'ensemble_method', 'heatmap')

        # Fast path: single run with no jitter — zero overhead vs old code.
        if n_runs <= 1 and jitter_scale == 0.0:
            masks, scores = predict_fn(image, boxes)
            return masks, scores, None

        all_masks = []   # list of (N, H, W) float32
        all_scores = []  # list of (N,) float32

        for _ in range(max(n_runs, 1)):
            run_boxes = self._jitter_boxes(boxes, jitter_scale) if jitter_scale > 0.0 else boxes
            masks, scores = predict_fn(image, run_boxes)
            if masks is None or len(masks) == 0:
                continue
            all_masks.append(masks.astype(np.float32))
            all_scores.append(scores.astype(np.float32))

        if not all_masks:
            return None, None, None

        if method == 'best_iou':
            # Keep the single run whose masks have the highest mean IoU score.
            mean_scores = [float(s.mean()) for s in all_scores]
            best = int(np.argmax(mean_scores))
            return all_masks[best].astype(np.uint8), all_scores[best], None

        # heatmap (default): average probability map → threshold at 0.5
        stacked = np.stack(all_masks, axis=0)          # (n_runs, N, H, W)
        prob_map = stacked.mean(axis=0)                 # (N, H, W) float32
        final_masks = (prob_map >= 0.5).astype(np.uint8)
        final_scores = np.stack(all_scores, axis=0).mean(axis=0)  # (N,)
        return final_masks, final_scores, prob_map

    @staticmethod
    def _save_tile_heatmap(
        heatmap: np.ndarray,
        tile_path: str,
        output_dir: 'Path',
    ) -> None:
        """Save a (H, W) float32 crown-probability heatmap as a georeferenced GeoTIFF.

        The spatial reference (transform + CRS) is copied from *tile_path* so
        the output raster overlays pixel-perfectly with the source tile.
        Pixel values range 0–1: fraction of ensemble runs that labelled the
        pixel as crown.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / Path(tile_path).name
        with rasterio.open(tile_path) as src:
            meta = src.meta.copy()
        meta.update(dtype='float32', count=1, nodata=None)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(heatmap[np.newaxis, :, :])

    def queue_masks(self,
                    box_object_ids: List[int or None],
                    masks: np.array,
                    image_size: Tuple[int, int],
                    scores: np.array,
                    tile_idx: int,
                    n_masks_processed: int,
                    queue: multiprocessing.JoinableQueue):
        
        # Scale down the masks to a fixed size to reduce memory footprint during postprocessing
        if self.config.pp_down_scale_masks_px and masks.shape[-1] > self.config.pp_down_scale_masks_px:
            resized_list = []
            for i in range(masks.shape[0]):
                mask = masks[i]
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8)

                mask_resized = cv2.resize(
                    mask,
                    (self.config.pp_down_scale_masks_px, self.config.pp_down_scale_masks_px),
                    interpolation=cv2.INTER_LINEAR
                )
                if masks.dtype == bool:
                    mask_resized = mask_resized > 0.5
                resized_list.append(mask_resized)

            # Stack them back along the batch dimension if you want a single tensor
            masks = np.stack(resized_list, axis=0)

        # Split masks and scores into chunks and put them into the queue for post-processing
        num_masks = masks.shape[0]
        chunk_size = max(1, num_masks // self.config.pp_n_workers)
        for j in range(0, num_masks, chunk_size):
            chunk_masks = masks[j:j + chunk_size]
            chunk_scores = scores[j:j + chunk_size]
            chunk_box_object_ids = box_object_ids[j:j + chunk_size]
            mask_ids = list(range(n_masks_processed, n_masks_processed + len(chunk_masks)))
            queue.put((tile_idx, mask_ids, chunk_box_object_ids, chunk_masks, chunk_scores, image_size))
            n_masks_processed += len(chunk_masks)

        return n_masks_processed

    def _load_ms_tile(self, tile_path: str, ms_tiles_path: str) -> Optional[np.ndarray]:
        """
        Load the multispectral tile that corresponds to the given RGB tile.

        The MS tile is expected to share the same *filename* as the RGB tile
        but reside in ``ms_tiles_path`` (or a matching sub-directory inside it).

        Args:
            tile_path: Absolute path to the RGB tile file.
            ms_tiles_path: Root directory of the MS tiles tree.

        Returns:
            CxHxW float32 numpy array, or ``None`` if no matching tile exists.
        """
        tile_name = Path(tile_path).name
        ms_tile_path = Path(ms_tiles_path) / tile_name

        if not ms_tile_path.exists():
            # Try recursive search for nested directory structures
            matches = list(Path(ms_tiles_path).rglob(tile_name))
            if not matches:
                return None
            ms_tile_path = matches[0]

        try:
            with rasterio.open(ms_tile_path) as src:
                return src.read().astype(np.float32)
        except Exception as exc:
            print(f"Warning: could not load MS tile '{ms_tile_path}': {exc}")
            return None

    def _infer_on_dataset(
        self,
        dataset: BaseDataset,
        collate_fn: object,
        ms_tiles_path: Optional[str] = None,
        heatmap_output_dir: Optional['Path'] = None,
    ):
        infer_dl = DataLoader(dataset, batch_size=self.config.image_batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)

        tiles_paths = []
        tiles_boxes_object_ids = []
        tiles_masks_polygons = []
        tiles_masks_scores = []
        queue = multiprocessing.JoinableQueue()  # Create a JoinableQueue

        # Store heatmap dir and tile paths as instance vars so forward() can access them
        # without changing the forward() signature across all subclasses.
        self._heatmap_output_dir = heatmap_output_dir
        self._pending_heatmap_tile_paths: List[str] = []

        print(f"Setting up {self.config.pp_n_workers} post-processing workers...")
        # Create a manager to share data across processes
        manager = multiprocessing.Manager()
        output_dict = manager.dict()
        processed_counter = multiprocessing.Value('i', 0)
        output_dict_lock = multiprocessing.Lock()

        # Start post-processing processes
        post_process_processes = []
        for _ in range(self.config.pp_n_workers):
            p = multiprocessing.Process(target=process_masks,
                                        args=(queue,
                                              output_dict,
                                              output_dict_lock,
                                              self.config.pp_simplify_tolerance,
                                              self.config.pp_remove_rings,
                                              self.config.pp_remove_small_geoms,
                                              processed_counter))
            p.start()
            post_process_processes.append(p)

        print("Post-processing workers are set up.")

        dataset_with_progress = tqdm(infer_dl,
                                     desc="Inferring the segmenter...",
                                     leave=True)                            # TODO check why its so slow here, like 30 seconds

        for i, sample in enumerate(dataset_with_progress):
            tiles_idx = list(range(i * self.config.image_batch_size, (i + 1) * self.config.image_batch_size))[:len(sample)]
            if isinstance(dataset, DetectionLabeledRasterCocoDataset):
                images, boxes, boxes_object_ids = sample
                current_tile_paths = [dataset.tiles[tile_idx]['path'] for tile_idx in tiles_idx]
                tiles_paths.extend(current_tile_paths)
            elif isinstance(dataset, UnlabeledRasterDataset):
                images = list(sample)
                boxes = [None] * len(images)
                boxes_object_ids = [None] * len(images)
                current_tile_paths = [dataset.tile_paths[tile_idx] for tile_idx in tiles_idx]
                tiles_paths.extend(current_tile_paths)
            else:
                raise ValueError("Dataset type not supported.")

            # Load corresponding MS tiles when ms_tiles_path is provided
            ms_images = None
            if ms_tiles_path is not None:
                ms_images = [
                    self._load_ms_tile(tp, ms_tiles_path)
                    for tp in current_tile_paths
                ]

            self._pending_heatmap_tile_paths = current_tile_paths
            self.forward(
                images=images,
                boxes=boxes,
                boxes_object_ids=boxes_object_ids,
                tiles_idx=tiles_idx,
                queue=queue,
                ms_images=ms_images,
            )

        print("Waiting for all postprocessing workers to be finished...")

        # Wait for all tasks in the queue to be completed
        queue.join()

        # Signal the end of input to the queue
        for _ in range(self.config.pp_n_workers):
            queue.put(None)

        # Wait for post-processing processes to finish
        for p in post_process_processes:
            p.join()

        # Close the queue
        queue.close()

        # Sorting the results within each tile_idx by mask_id to maintain order
        for tile_idx in output_dict.keys():
            output_dict[tile_idx] = sorted(output_dict[tile_idx], key=lambda x: x[0])

        # Assemble the results into tiles_masks_polygons
        for tile_idx in sorted(output_dict.keys()):
            _, box_object_ids, masks_polygons, scores = zip(*output_dict[tile_idx])
            box_object_ids = list(box_object_ids)
            masks_polygons = list(masks_polygons)
            scores = [score.item() for score in scores]

            tiles_boxes_object_ids.append(box_object_ids)
            tiles_masks_polygons.append(masks_polygons)
            tiles_masks_scores.append(scores)

        print(f"Finished inferring the segmenter {self.config.model}-{self.config.architecture}.")

        if isinstance(dataset, UnlabeledRasterDataset):
            # There were no box prompts, so we return None for the boxes_object_ids instead of lists of None values
            tiles_boxes_object_ids = None

        return tiles_paths, tiles_boxes_object_ids, tiles_masks_polygons, tiles_masks_scores
