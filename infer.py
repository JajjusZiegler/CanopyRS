import argparse
import logging
import shutil
import warnings
from pathlib import Path
from warnings import warn

from canopyrs.engine.utils import init_spawn_method

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing from timm.models.layers is deprecated"
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)
detrex_logger = logging.getLogger("detrex.checkpoint.c2_model_loading")
detrex_logger.disabled = True

from canopyrs.engine.config_parsers import InferIOConfig, PipelineConfig
from canopyrs.engine.config_parsers.base import get_config_path
from canopyrs.engine.pipeline import Pipeline


def pipeline_main(args: argparse.Namespace) -> None:
    config_path = get_config_path(f'{args.config}')
    config = PipelineConfig.from_yaml(config_path)

    if args.io_config_path and (args.imagery_path or args.output_path):
        raise ValueError("Either provide an io config file or pass imagery/tiles path and output path as arguments.")
    elif args.io_config_path:
        io_config = InferIOConfig.from_yaml(args.io_config_path)
        component_ids = None  # use default sequential IDs
    elif (args.imagery_path or args.tiles_path) and args.output_path:
        # Track the original index of each component so folders always get their
        # canonical IDs (0_tilerizer, 1_detector … 3_segmenter, 4_aggregator)
        # even when leading/middle components are skipped for a resume run.
        component_ids = list(range(len(config.components_configs)))

        # Remove tilerizer if tiles are already provided (tilerizer not needed).
        if args.tiles_path and args.imagery_path is None and config.components_configs[0][0] == 'tilerizer':
            warn('Removing the first component (tilerizer) from the pipeline as it is not needed, since tiles are already provided as input.')
            config.components_configs.pop(0)
            component_ids.pop(0)

        # Resume from aggregated detections: skip components already completed.
        if args.resume_from_gpkg:
            if not args.input_coco:
                raise ValueError("--resume-from-gpkg requires --input-coco to be provided (COCO JSON from 2_aggregator/ or 1_detector/).")

            if args.tiles_path:
                # Tiles exist: skip every component before the segmenter.
                while config.components_configs and config.components_configs[0][0] != 'segmenter':
                    warn(f"Resuming: skipping completed component '{config.components_configs[0][0]}'")
                    config.components_configs.pop(0)
                    component_ids.pop(0)
            else:
                # Tiles were deleted but aggregated detections exist: keep the
                # tilerizer so tiles are recreated, then skip detector + first
                # aggregator and jump straight to the segmenter.
                seg_pos = next(
                    (i for i, (t, _) in enumerate(config.components_configs) if t == 'segmenter'),
                    None
                )
                if seg_pos is None:
                    raise ValueError("No 'segmenter' component found in pipeline config — cannot resume.")
                # Remove everything between tilerizer (index 0) and segmenter.
                for _ in range(seg_pos - 1):
                    warn(f"Resuming (re-tilerize): skipping completed component '{config.components_configs[1][0]}'")
                    config.components_configs.pop(1)
                    component_ids.pop(1)

            if not config.components_configs or config.components_configs[0][0] not in ('segmenter', 'tilerizer'):
                raise ValueError("No 'segmenter' component found in pipeline config — cannot resume.")

        config_args = {
            'input_imagery': args.imagery_path,
            'output_folder': args.output_path,
            'tiles_path': args.tiles_path
        }
        if args.aoi_path:
            config_args['aoi_config'] = 'package'
            config_args['aoi'] = args.aoi_path
        if args.multispectral:
            config_args['multispectral_imagery'] = args.multispectral
        if args.resume_from_gpkg:
            config_args['input_gpkg'] = args.resume_from_gpkg
        if args.input_coco:
            config_args['input_coco'] = args.input_coco

        io_config = InferIOConfig(**config_args)
    else:
        raise ValueError("Either provide an io config file or pass imagery/tiles path and output path as arguments.")

    pipeline = Pipeline.from_config(io_config, config, component_ids=component_ids)
    data_state = pipeline.run(strict_rgb_validation=not args.no_strict_rgb)

    if args.delete_tiles:
        for attr in ("tiles_path", "ms_tiles_path"):
            tile_dir = getattr(data_state, attr, None)
            if tile_dir and Path(tile_dir).exists():
                shutil.rmtree(tile_dir)
                print(f"Deleted tiles: {tile_dir}")


if __name__ == '__main__':
    init_spawn_method()
    parser = argparse.ArgumentParser()

    # Inference args
    parser.add_argument("-c", "--config", type=str, default='default', help="Name of a default, predefined config or path to the appropriate .yaml config file.")
    parser.add_argument("-io", "--io_config_path", type=str, help="Path to the appropriate .yaml io config file.")
    parser.add_argument("-i", "--imagery_path", type=str, help="Path to the imagery.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output folder.")
    parser.add_argument("-t", "--tiles_path", type=str, help="Path to the tiles folder to infer on.")
    parser.add_argument("-aoi", "--aoi_path", type=str, help="Path to the area of interest (AOI) geopackage.")
    parser.add_argument("-ms", "--multispectral", type=str, default=None,
                        help="Path to a co-registered multispectral raster (enables dual-stream MS inference).")
    parser.add_argument("--no-strict-rgb", action="store_true", default=False,
                        help="Disable strict RGB band validation (use for multispectral rasters).")
    parser.add_argument("--delete-tiles", action="store_true", default=False,
                        help="Delete the tile directory (RGB and MS) after the pipeline finishes successfully.")
    parser.add_argument("--resume-from-gpkg", type=str, default=None,
                        help="Path to an existing aggregated-detections GeoPackage (e.g. from 2_aggregator/). "
                             "Skips all pipeline components before the segmenter (tilerizer, detector, "
                             "pre-segmenter aggregator). Requires --tiles-path and --input-coco.")
    parser.add_argument("--input-coco", type=str, default=None,
                        help="Path to an existing COCO JSON file (e.g. from 2_aggregator/ or 1_detector/). "
                             "Used together with --resume-from-gpkg to seed infer_coco_path in the pipeline "
                             "so the segmenter can read bounding-box prompts without re-running detection.")

    args = parser.parse_args()

    pipeline_main(args)




