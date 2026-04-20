from typing import Literal

from canopyrs.engine.config_parsers.base import BaseConfig


class MultiRunMergerConfig(BaseConfig):
    """
    Configuration for MultiRunMergerComponent.

    Attributes:
        iou_threshold: Minimum IoU to consider two polygons as representing the same tree.
                       Polygons from different runs with IoU >= this value are clustered together.
                       Default 0.3 is intentionally permissive for organic crown shapes.
        min_observations: Minimum number of runs that must have detected a tree for it to be
                          included in the output. 1 = keep all, 2 = keep only trees seen by >=2 runs.
        geometry_method: How to compute the consensus geometry.
                         - 'majority_union': union of polygons seen in >= ceil(n_runs * majority_threshold) runs
                         - 'intersection_union_blend': weighted blend between intersection and union
                         - 'best_score': simply keep the polygon with the highest segmenter_score
        majority_threshold: Fraction of runs that must agree for 'majority_union' geometry method.
                            Ignored for other geometry methods. Default 0.4 (40% of runs must agree).
        score_aggregation: How to compute the consensus score from the cluster's scores.
                           'mean' | 'max' | 'weighted_mean' (weighted by polygon area).
        add_observation_count: Whether to add an 'observation_count' column to the output GDF.
        add_run_agreement: Whether to add a 'run_agreement' column (IoU of union vs intersection).
        add_source_runs: Whether to add a 'source_runs' column listing which run indices detected each tree.
    """

    iou_threshold: float = 0.3
    min_observations: int = 1
    geometry_method: Literal['majority_union', 'intersection_union_blend', 'best_score'] = 'majority_union'
    majority_threshold: float = 0.4
    score_aggregation: Literal['mean', 'max', 'weighted_mean'] = 'mean'
    add_observation_count: bool = True
    add_run_agreement: bool = True
    add_source_runs: bool = True
