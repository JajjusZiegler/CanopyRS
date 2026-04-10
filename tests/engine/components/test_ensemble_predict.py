"""
Unit tests for _jitter_boxes() and _ensemble_predict() in SegmenterWrapperBase.

These tests use a lightweight stub — no torch, cv2, or SAM model required.
"""

import types
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal stub that exposes only the two new helpers so we can test them
# without loading the full segmenter_base module (which needs cv2/torch).
# ---------------------------------------------------------------------------

def _jitter_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    """Copied verbatim from segmenter_base.SegmenterWrapperBase._jitter_boxes."""
    boxes = boxes.copy().astype(np.float32)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    boxes[:, 0] += np.random.uniform(-scale, scale, len(boxes)) * widths
    boxes[:, 1] += np.random.uniform(-scale, scale, len(boxes)) * heights
    boxes[:, 2] += np.random.uniform(-scale, scale, len(boxes)) * widths
    boxes[:, 3] += np.random.uniform(-scale, scale, len(boxes)) * heights
    return boxes


def _make_config(n_runs=1, jitter=0.0, method="heatmap"):
    cfg = types.SimpleNamespace(
        ensemble_n_runs=n_runs,
        ensemble_box_jitter_scale=jitter,
        ensemble_method=method,
    )
    return cfg


def _ensemble_predict(config, predict_fn, image, boxes):
    """
    Thin re-implementation of segmenter_base._ensemble_predict for isolated
    testing — keeps identical semantics.
    """
    n_runs = getattr(config, 'ensemble_n_runs', 1)
    jitter_scale = getattr(config, 'ensemble_box_jitter_scale', 0.0)
    method = getattr(config, 'ensemble_method', 'heatmap')

    # Fast path
    if n_runs <= 1 and jitter_scale == 0.0:
        return predict_fn(image, boxes)

    all_masks = []
    all_scores = []

    for _ in range(max(n_runs, 1)):
        run_boxes = _jitter_boxes(boxes, jitter_scale) if jitter_scale > 0.0 else boxes
        masks, scores = predict_fn(image, run_boxes)
        if masks is None or len(masks) == 0:
            continue
        all_masks.append(masks.astype(np.float32))
        all_scores.append(scores.astype(np.float32))

    if not all_masks:
        return None, None

    if method == 'best_iou':
        mean_scores = [float(s.mean()) for s in all_scores]
        best = int(np.argmax(mean_scores))
        return all_masks[best].astype(np.uint8), all_scores[best]

    # heatmap
    stacked = np.stack(all_masks, axis=0)
    prob_map = stacked.mean(axis=0)
    final_masks = (prob_map >= 0.5).astype(np.uint8)
    final_scores = np.stack(all_scores, axis=0).mean(axis=0)
    return final_masks, final_scores


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _make_predict_fn(masks, scores):
    """Return a predict_fn that always returns the given masks & scores."""
    call_count = [0]

    def fn(image, boxes):
        call_count[0] += 1
        return masks.copy(), scores.copy()

    fn.call_count = call_count
    return fn


def _constant_masks(n=3, h=8, w=8, value=1):
    masks = np.full((n, h, w), value, dtype=np.uint8)
    scores = np.ones(n, dtype=np.float32)
    return masks, scores


# ===========================================================================
# Tests: _jitter_boxes
# ===========================================================================

class TestJitterBoxes:
    def test_output_shape_preserved(self):
        boxes = np.array([[10, 20, 50, 60], [5, 5, 15, 25]], dtype=np.float32)
        out = _jitter_boxes(boxes, 0.05)
        assert out.shape == boxes.shape

    def test_zero_scale_returns_original_values(self):
        boxes = np.array([[10, 20, 50, 60]], dtype=np.float32)
        out = _jitter_boxes(boxes, 0.0)
        np.testing.assert_array_equal(out, boxes)

    def test_jitter_stays_within_expected_range(self):
        np.random.seed(0)
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scale = 0.1
        for _ in range(200):
            out = _jitter_boxes(boxes, scale)
            # Each corner can shift by at most ±scale * side_length = ±10
            assert out[0, 0] >= 100 - 10 - 1e-4
            assert out[0, 0] <= 100 + 10 + 1e-4
            assert out[0, 2] >= 200 - 10 - 1e-4
            assert out[0, 2] <= 200 + 10 + 1e-4

    def test_does_not_modify_original(self):
        boxes = np.array([[10, 20, 50, 60]], dtype=np.float32)
        original = boxes.copy()
        _jitter_boxes(boxes, 0.1)
        np.testing.assert_array_equal(boxes, original)

    def test_output_dtype_float32(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.int32)
        out = _jitter_boxes(boxes, 0.05)
        assert out.dtype == np.float32


# ===========================================================================
# Tests: _ensemble_predict — fast path (n_runs=1, jitter=0)
# ===========================================================================

class TestEnsemblePredictFastPath:
    def test_single_run_calls_predict_fn_once(self):
        masks, scores = _constant_masks()
        fn = _make_predict_fn(masks, scores)
        cfg = _make_config(n_runs=1, jitter=0.0)
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

        _ensemble_predict(cfg, fn, None, boxes)
        assert fn.call_count[0] == 1

    def test_single_run_returns_predict_fn_output_directly(self):
        masks, scores = _constant_masks()
        fn = _make_predict_fn(masks, scores)
        cfg = _make_config(n_runs=1, jitter=0.0)
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

        out_masks, out_scores = _ensemble_predict(cfg, fn, None, boxes)
        np.testing.assert_array_equal(out_masks, masks)
        np.testing.assert_array_equal(out_scores, scores)


# ===========================================================================
# Tests: _ensemble_predict — heatmap method
# ===========================================================================

class TestEnsemblePredictHeatmap:
    def test_n_runs_controls_call_count(self):
        masks, scores = _constant_masks()
        fn = _make_predict_fn(masks, scores)
        cfg = _make_config(n_runs=5, jitter=0.0, method="heatmap")
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

        _ensemble_predict(cfg, fn, None, boxes)
        assert fn.call_count[0] == 5

    def test_all_ones_heatmap_stays_ones(self):
        """If every run returns all-1 masks, heatmap result should also be all 1."""
        masks, scores = _constant_masks(n=3, value=1)
        fn = _make_predict_fn(masks, scores)
        cfg = _make_config(n_runs=4, jitter=0.0, method="heatmap")
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

        out_masks, out_scores = _ensemble_predict(cfg, fn, None, boxes)
        assert out_masks.dtype == np.uint8
        assert out_masks.max() == 1
        np.testing.assert_array_equal(out_masks, masks)

    def test_all_zeros_heatmap_stays_zeros(self):
        masks, scores = _constant_masks(n=3, value=0)
        fn = _make_predict_fn(masks, scores)
        cfg = _make_config(n_runs=3, jitter=0.0, method="heatmap")
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

        out_masks, out_scores = _ensemble_predict(cfg, fn, None, boxes)
        assert out_masks.max() == 0

    def test_majority_vote_threshold(self):
        """
        3 runs: 2 return 1, 1 returns 0 → average 0.67 → thresholded to 1.
        2 runs: 2 return 0, 1 returns 1 → average 0.33 → thresholded to 0.
        """
        h, w = 4, 4
        n = 2  # 2 crowns

        call_count = [0]

        def alternating_fn(image, boxes):
            call_count[0] += 1
            if call_count[0] <= 2:
                return np.ones((n, h, w), dtype=np.uint8), np.ones(n, dtype=np.float32)
            else:
                return np.zeros((n, h, w), dtype=np.uint8), np.ones(n, dtype=np.float32)

        cfg = _make_config(n_runs=3, jitter=0.0, method="heatmap")
        boxes = np.zeros((n, 4), dtype=np.float32)

        out_masks, _ = _ensemble_predict(cfg, alternating_fn, None, boxes)
        # 2 out of 3 runs returned 1 → prob=0.67 → above 0.5 → mask=1
        assert out_masks.max() == 1
        assert out_masks.min() == 1

    def test_output_shape_matches_input_crowns(self):
        n, h, w = 5, 16, 16
        masks = np.ones((n, h, w), dtype=np.uint8)
        scores = np.ones(n, dtype=np.float32)
        fn = _make_predict_fn(masks, scores)
        cfg = _make_config(n_runs=3, jitter=0.0, method="heatmap")
        boxes = np.zeros((n, 4), dtype=np.float32)

        out_masks, out_scores = _ensemble_predict(cfg, fn, None, boxes)
        assert out_masks.shape == (n, h, w)
        assert out_scores.shape == (n,)

    def test_scores_are_mean_across_runs(self):
        n, h, w = 2, 4, 4
        call_count = [0]
        score_values = [0.9, 0.7, 0.5]

        def varying_score_fn(image, boxes):
            s = score_values[call_count[0]]
            call_count[0] += 1
            return np.ones((n, h, w), dtype=np.uint8), np.full(n, s, dtype=np.float32)

        cfg = _make_config(n_runs=3, jitter=0.0, method="heatmap")
        boxes = np.zeros((n, 4), dtype=np.float32)

        _, out_scores = _ensemble_predict(cfg, varying_score_fn, None, boxes)
        expected = np.mean(score_values)
        np.testing.assert_allclose(out_scores, expected, atol=1e-5)


# ===========================================================================
# Tests: _ensemble_predict — best_iou method
# ===========================================================================

class TestEnsemblePredictBestIou:
    def test_returns_run_with_highest_mean_score(self):
        n, h, w = 2, 4, 4
        call_count = [0]
        # Run 0: score=0.3 (bad), Run 1: score=0.9 (good), Run 2: score=0.6 (ok)
        score_values = [0.3, 0.9, 0.6]
        mask_values = [0, 1, 0]  # marks which run to identify

        def fn(image, boxes):
            i = call_count[0]
            call_count[0] += 1
            m = np.full((n, h, w), mask_values[i], dtype=np.uint8)
            s = np.full(n, score_values[i], dtype=np.float32)
            return m, s

        cfg = _make_config(n_runs=3, jitter=0.0, method="best_iou")
        boxes = np.zeros((n, 4), dtype=np.float32)

        out_masks, out_scores = _ensemble_predict(cfg, fn, None, boxes)
        # Run 1 has highest score (0.9) → mask value should be 1
        assert out_masks.max() == 1
        np.testing.assert_allclose(out_scores[0], 0.9, atol=1e-5)

    def test_best_iou_does_not_average_masks(self):
        """best_iou should return binary masks, not an average."""
        n, h, w = 2, 4, 4

        def fn(image, boxes):
            # Always return 1-masks
            return np.ones((n, h, w), dtype=np.uint8), np.ones(n, dtype=np.float32)

        cfg = _make_config(n_runs=3, jitter=0.0, method="best_iou")
        boxes = np.zeros((n, 4), dtype=np.float32)

        out_masks, _ = _ensemble_predict(cfg, fn, None, boxes)
        assert out_masks.dtype == np.uint8
        assert set(np.unique(out_masks)).issubset({0, 1})


# ===========================================================================
# Tests: edge cases
# ===========================================================================

class TestEnsemblePredictEdgeCases:
    def test_predict_fn_returning_none_returns_none_none(self):
        def fn(image, boxes):
            return None, None

        cfg = _make_config(n_runs=3, jitter=0.05, method="heatmap")
        boxes = np.zeros((2, 4), dtype=np.float32)

        out = _ensemble_predict(cfg, fn, None, boxes)
        assert out == (None, None)

    def test_predict_fn_returning_empty_returns_none_none(self):
        def fn(image, boxes):
            return np.array([]), np.array([])

        cfg = _make_config(n_runs=3, jitter=0.05, method="heatmap")
        boxes = np.zeros((2, 4), dtype=np.float32)

        out = _ensemble_predict(cfg, fn, None, boxes)
        assert out == (None, None)

    def test_jitter_enabled_changes_boxes_each_run(self):
        """With jitter, the boxes passed to predict_fn should differ across runs."""
        received_boxes = []

        def fn(image, boxes):
            received_boxes.append(boxes.copy())
            n = len(boxes)
            return np.ones((n, 4, 4), dtype=np.uint8), np.ones(n, dtype=np.float32)

        np.random.seed(42)
        cfg = _make_config(n_runs=5, jitter=0.1, method="heatmap")
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)

        _ensemble_predict(cfg, fn, None, boxes)
        assert len(received_boxes) == 5
        # Not all boxes should be identical
        all_same = all(np.allclose(received_boxes[0], b) for b in received_boxes[1:])
        assert not all_same

    def test_n_runs_zero_treated_as_one(self):
        """n_runs=0 should still call predict_fn (max(n_runs, 1) guard)."""
        masks, scores = _constant_masks()
        fn = _make_predict_fn(masks, scores)
        cfg = _make_config(n_runs=0, jitter=0.05, method="heatmap")
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

        out_masks, _ = _ensemble_predict(cfg, fn, None, boxes)
        assert out_masks is not None
        assert fn.call_count[0] == 1


# ===========================================================================
# Tests: SegmenterConfig new fields
# ===========================================================================

class TestSegmenterConfigEnsembleFields:
    def test_default_values_preserve_single_pass(self):
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig()
        assert cfg.ensemble_n_runs == 1
        assert cfg.ensemble_box_jitter_scale == 0.0
        assert cfg.ensemble_method == "heatmap"

    def test_can_set_ensemble_fields(self):
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig(ensemble_n_runs=5, ensemble_box_jitter_scale=0.05, ensemble_method="best_iou")
        assert cfg.ensemble_n_runs == 5
        assert cfg.ensemble_box_jitter_scale == 0.05
        assert cfg.ensemble_method == "best_iou"

    def test_from_dict_with_ensemble_fields(self):
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig.from_dict({
            "ensemble_n_runs": 3,
            "ensemble_box_jitter_scale": 0.03,
            "ensemble_method": "heatmap",
        })
        assert cfg.ensemble_n_runs == 3
        assert cfg.ensemble_box_jitter_scale == 0.03
