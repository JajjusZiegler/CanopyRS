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
    elif (args.imagery_path or args.tiles_path) and args.output_path:
        # Check if the first component is 'tilerizer' and remove it if tiles are provided (i.e. tilerizer is not needed).
        if args.tiles_path and args.imagery_path is None and config.components_configs[0][0] == 'tilerizer':
            warn('Removing the first component (tilerizer) from the pipeline as it is not needed, since tiles are already provided as input.')
            config.components_configs.pop(0)

        # Resume from aggregated detections: skip all leading components up to the segmenter.
        # Useful when tilerization + detection + first aggregation already completed in a
        # previous run and only the segmentation (and final aggregation) need to be re-run.
        if args.resume_from_gpkg:
            if not args.tiles_path:
                raise ValueError("--resume-from-gpkg requires --tiles-path to be provided.")
            if not args.input_coco:
                raise ValueError("--resume-from-gpkg requires --input-coco to be provided (COCO JSON from 2_aggregator/ or 1_detector/).")
            while config.components_configs and config.components_configs[0][0] != 'segmenter':
                warn(f"Resuming: skipping completed component '{config.components_configs[0][0]}'")
                config.components_configs.pop(0)
            if not config.components_configs:
                raise ValueError(
                    "No 'segmenter' component found in pipeline config — "
                    "cannot resume from stage 2. Check your config."
                )

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

    pipeline = Pipeline.from_config(io_config, config)
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




