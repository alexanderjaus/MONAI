from __future__ import annotations

import gc
import hashlib
import os
from typing import Any, Sequence

import numpy as np
import torch

from monai.metrics import (
    Cumulative,
    CumulativeIterationMetric,
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)
from monai.utils.module import optional_import

__all__ = [
    "CCBaseMetric",
    "CCDiceMetric",
    "CCHausdorffDistanceMetric",
    "CCHausdorffDistance95Metric",
    "CCSurfaceDistanceMetric",
    "CCSurfaceDiceMetric",
]


# Optional SciPy imports (CPU path)
distance_transform_edt, _has_scipy_edt = optional_import("scipy.ndimage", name="distance_transform_edt")
sn_label, _has_scipy_label = optional_import("scipy.ndimage", name="label")
generate_binary_structure, _has_scipy_struct = optional_import(
    "scipy.ndimage", name="generate_binary_structure"
)


def _compute_voronoi_regions_fast_cpu(labels: np.ndarray, connectivity: int = 26, sampling: Sequence[float] | None = None) -> np.ndarray:
    """
    Compute Voronoi assignment to connected components (CPU, single EDT) without external dependencies.
    Voxels with labels>0 are seeds. Returns, for each voxel, the ID of the nearest component tag.

    Args:
        labels: 3D integer array. Non-zero voxels indicate foreground.
        connectivity: 6/18/26 (3D) neighborhood definition.
        sampling: voxel spacing for anisotropic distances, forwarded to SciPy EDT.
    """
    if not (_has_scipy_edt and _has_scipy_label and _has_scipy_struct):
        # Defer error until actual use to align with MONAI optional dependency behavior
        raise RuntimeError(
            "SciPy is required for CC-Metrics Voronoi preprocessing (ndimage.label/distance_transform_edt)."
        )

    x = np.asarray(labels)
    # Map 3D connectivity to SciPy structure connectivity
    conn_rank = {6: 1, 18: 2, 26: 3}.get(connectivity, 3)
    structure = generate_binary_structure(rank=3, connectivity=conn_rank)
    cc, num = sn_label(x > 0, structure=structure)

    if num == 0:
        return np.zeros_like(x, dtype=np.int32)

    # EDT input: 0 at seeds, 1 elsewhere
    edt_input = np.ones(cc.shape, dtype=np.uint8)
    edt_input[cc > 0] = 0

    # Get indices of the nearest seeds (no distances array needed)
    indices = distance_transform_edt(
        edt_input, sampling=sampling, return_distances=False, return_indices=True
    )

    voronoi = cc[tuple(indices)]  # component tag at nearest seed
    return voronoi.astype(np.int32, copy=False)


class CCBaseMetric:
    """
    Connected-Component-aware wrapper around a MONAI cumulative metric.

    It partitions the ground-truth foreground into connected components, crops a local ROI per component,
    remaps those ROIs to one-hot representation, and evaluates the provided base metric inside each ROI.
    The per-component scores are collected in an internal buffer and can be aggregated per-patient or overall.

    Constraints:
    - Binary segmentation only (2 channels, foreground/background)
    - Batch size 1 (B=1)
    - Background is always excluded (include_background=False)
    """

    def __init__(
        self,
        BaseMetric: type[Cumulative] | type[CumulativeIterationMetric],
        *args: Any,
        use_caching: bool = False,
        caching_dir: str = ".cache",
        metric_best_score: float | None = None,
        metric_worst_score: float | None = None,
        cc_reduction: str | None = None,
        **kwargs: Any,
    ) -> None:
        assert metric_best_score is not None, "Best score must be defined"
        assert metric_worst_score is not None, "Worst score must be defined"

        # Background must be excluded for component-wise evaluation
        if kwargs.get("include_background", False):
            raise ValueError("include_background=True is not supported for CC-Metrics")
        kwargs["include_background"] = False

        if cc_reduction is None:
            cc_reduction = "patient"
        if cc_reduction not in ("patient", "overall"):
            raise ValueError(f"Unknown cc_reduction: {cc_reduction}")
        self.cc_reduction = cc_reduction

        self.metric_perfect_score = metric_best_score
        self.metric_worst_score = metric_worst_score
        self.base_metric = BaseMetric(*args, **kwargs)

        self.use_caching = use_caching
        self.caching_dir = caching_dir
        if self.use_caching and not os.path.exists(self.caching_dir):
            os.makedirs(self.caching_dir)

        # CPU backend by default
        self.xp = np
        self.backend = "numpy"
        self._space_separation = _compute_voronoi_regions_fast_cpu

        self._buffer_collection: list[torch.Tensor] = []

    def _verify_and_convert(self, y_pred: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        # Convert incoming tensors to numpy
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if not isinstance(y_pred, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("y_pred and y must be numpy arrays or torch tensors")

        if len(y_pred.shape) != 5:
            raise AssertionError("Expected y_pred shape (B,C,D,H,W)")
        if len(y.shape) != 5:
            raise AssertionError("Expected y shape (B,C,D,H,W)")
        if y_pred.shape != y.shape:
            raise AssertionError(f"Input shapes do not match: {y_pred.shape} vs {y.shape}")
        if y_pred.shape[1] != 2 or y.shape[1] != 2:
            raise AssertionError(f"Expected 2 classes (binary). Got {y_pred.shape[1]}")
        if y_pred.shape[0] != 1 or y.shape[0] != 1:
            raise AssertionError("Only batch size of 1 is supported")

        return y_pred, y

    def _convert_to_target(self, y_pred: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(y_pred), torch.from_numpy(y)

    def __call__(self, y_pred: Any, y: Any) -> None:
        y_pred, y = self._verify_and_convert(y_pred, y)

        # Compute argmax channel-wise
        pred_helper = y_pred.argmax(1)
        label_helper = y.argmax(1)

        # Background-only label special-cases
        if label_helper[0].sum() == 0:
            if pred_helper[0].sum() == 0:
                self._buffer_collection.append(torch.tensor([self.metric_perfect_score]))
            else:
                self._buffer_collection.append(torch.tensor([self.metric_worst_score]))
            return

        # Component assignment on GT foreground
        cc_assignment = self._space_separation(label_helper[0])

        # Evaluate per component
        for cc_id in self.xp.unique(cc_assignment):
            if cc_id == 0:
                # zero is background in labeled array; skip
                continue
            cc_mask = cc_assignment == cc_id
            coords = self.xp.argwhere(cc_mask)
            if coords.size == 0:
                continue
            min_corner_idx = coords.min(axis=0)
            max_corner_idx = coords.max(axis=0)

            # Crop ROI around component
            z0, y0, x0 = min_corner_idx
            z1, y1, x1 = max_corner_idx + 1
            crop_pred = pred_helper[0][z0:z1, y0:y1, x0:x1]
            crop_label = label_helper[0][z0:z1, y0:y1, x0:x1]

            # Mask to component extent
            cc_roi = cc_mask[z0:z1, y0:y1, x0:x1]
            pred_masked = crop_pred * cc_roi
            label_masked = crop_label * cc_roi

            # Convert to one-hot in numpy and then to torch CPU tensors
            pred_onehot = self.xp.moveaxis(self.xp.eye(2, dtype=self.xp.uint8)[pred_masked], -1, 0)
            label_onehot = self.xp.moveaxis(self.xp.eye(2, dtype=self.xp.uint8)[label_masked], -1, 0)
            pred_onehot_t, label_onehot_t = self._convert_to_target(
                pred_onehot[self.xp.newaxis], label_onehot[self.xp.newaxis]
            )

            # Delegate to base MONAI metric
            self.base_metric(y_pred=pred_onehot_t, y=label_onehot_t)

            # free intermediates
            del crop_pred, crop_label, pred_masked, label_masked, cc_mask
            gc.collect()

        # Retrieve and reset the base metric buffer for this case
        metric_buffer = self.base_metric.get_buffer()
        # base metrics typically have a single tensor buffer
        if isinstance(metric_buffer, list):
            # if a list of buffers, flatten/stack relevant parts; default to first
            metric_tensor = metric_buffer[0]
        else:
            metric_tensor = metric_buffer
        self._buffer_collection.append(metric_tensor)
        self.base_metric.reset()

    def cc_aggregate(self, mode: str | None = None) -> torch.Tensor:
        """
        Aggregates collected per-component scores.

        - "patient": mean per case (returns per-case mean as 1D tensor)
        - "overall": all components equally (returns flat 1D tensor of all components)
        """
        if mode is None:
            mode = self.cc_reduction
        if mode not in ("patient", "overall"):
            raise ValueError(f"Unknown aggregation mode: {mode}")

        cleaned: list[torch.Tensor] = []
        for x in self._buffer_collection:
            # Replace inf/nan with worst score
            x = torch.where(torch.isinf(x), torch.tensor(self.metric_worst_score, dtype=torch.float32, device=x.device), x)
            x = torch.where(torch.isnan(x), torch.tensor(self.metric_worst_score, dtype=torch.float32, device=x.device), x)
            cleaned.append(x.reshape(-1, 1))

        if not cleaned:
            return torch.empty(0, dtype=torch.float32)

        if mode == "patient":
            return torch.stack([t.mean() for t in cleaned])
        # overall
        return torch.cat(cleaned, dim=0).squeeze()

    # MONAI-style API compatibility
    def aggregate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        mode = kwargs.pop("mode", None)
        return self.cc_aggregate(mode=mode)

    def get_buffer(self) -> list[torch.Tensor]:
        return self._buffer_collection

    def reset(self) -> None:
        self._buffer_collection = []
        # do not reset the base metric here; done per-case in __call__

    # Optional caching utility (precompute connected components from GT)
    def cache_datapoint(self, y: torch.Tensor | np.ndarray) -> None:
        if not self.use_caching:
            raise ValueError("Caching is disabled")
        arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input must be numpy array or torch tensor")
        if arr.ndim != 3:
            raise AssertionError("Expected (D,H,W) array for caching")
        gt_fingerprint = hashlib.md5(arr.tobytes()).hexdigest()
        target_path = os.path.join(self.caching_dir, f"{gt_fingerprint}.npy")
        if os.path.exists(target_path):
            return
        cc_assignment = self._space_separation(arr)
        np.save(target_path, cc_assignment)


class CCDiceMetric(CCBaseMetric):
    """Connected-component Dice based on MONAI DiceMetric (best=1.0, worst=0.0)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(DiceMetric, *args, metric_best_score=1.0, metric_worst_score=0.0, **kwargs)

    def __call__(self, y_pred: Any, y: Any) -> None:  # optimized bincount variant
        y_pred, y = self._verify_and_convert(y_pred, y)
        pred_helper = y_pred.argmax(1)
        label_helper = y.argmax(1)

        if label_helper[0].sum() == 0:
            if pred_helper[0].sum() == 0:
                self.get_buffer().append(torch.tensor([self.metric_perfect_score]))
            else:
                self.get_buffer().append(torch.tensor([self.metric_worst_score]))
            return

        cc_assignment = self._space_separation(label_helper[0])

        uniq, inv = self.xp.unique(cc_assignment.ravel(), return_inverse=True)
        # 0 may be background from labeling; keep all unique and compute stats
        nof_components = uniq.size

        code = (label_helper.ravel() << 1) | pred_helper.ravel()
        idx = (inv << 2) | code
        hist = self.xp.bincount(idx, minlength=nof_components * 4).reshape(-1, 4)
        TN, FP, FN, TP = hist[:, 0], hist[:, 1], hist[:, 2], hist[:, 3]
        denom = 2 * TP + FP + FN
        dice_scores = self.xp.where(denom > 0, (2 * TP) / denom, 1.0)
        dice_scores_t = (
            torch.from_numpy(self.xp.asnumpy(dice_scores)) if self.backend == "cupy" else torch.from_numpy(dice_scores)
        )
        self.get_buffer().append(dice_scores_t.unsqueeze(-1))


class CCHausdorffDistanceMetric(CCBaseMetric):
    """Connected-component 100% Hausdorff distance using MONAI HausdorffDistanceMetric (best=0.0)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(HausdorffDistanceMetric, *args, metric_best_score=0.0, metric_worst_score=float("inf"), **kwargs)


class CCHausdorffDistance95Metric(CCBaseMetric):
    """Connected-component 95% Hausdorff distance using MONAI HausdorffDistanceMetric (best=0.0)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            HausdorffDistanceMetric, *args, metric_best_score=0.0, metric_worst_score=float("inf"), percentile=95, **kwargs
        )


class CCSurfaceDistanceMetric(CCBaseMetric):
    """Connected-component surface distance using MONAI SurfaceDistanceMetric (best=0.0)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(SurfaceDistanceMetric, *args, metric_best_score=0.0, metric_worst_score=float("inf"), **kwargs)


class CCSurfaceDiceMetric(CCBaseMetric):
    """Connected-component surface dice using MONAI SurfaceDiceMetric (best=1.0, worst=0.0)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            SurfaceDiceMetric, *args, metric_best_score=1.0, metric_worst_score=0.0, **kwargs
        )


# Optional GPU variants using CuPy/cupyx
cp, _has_cupy = optional_import("cupy")
cupy_ndimage, _has_cupyx = optional_import("cupyx.scipy.ndimage")


def _compute_voronoi_regions_fast_gpu(labels: Any, connectivity: int = 26, sampling: Sequence[float] | None = None, return_numpy: bool = False):
    if not (_has_cupy and _has_cupyx):
        raise RuntimeError("cupy and cupyx.scipy.ndimage are required for GPU CC-Metrics")

    rank = {6: 1, 18: 2, 26: 3}.get(connectivity, 3)
    x = cp.asarray(labels)
    if (x > 0).sum() == 0:
        out = cp.zeros_like(x, dtype=cp.int32)
        return cp.asnumpy(out) if return_numpy else out

    structure = cupy_ndimage.generate_binary_structure(rank=3, connectivity=rank)
    cc, num = cupy_ndimage.label(x > 0, structure=structure)
    if num == 0:
        out = cp.zeros_like(x, dtype=cp.int32)
        return cp.asnumpy(out) if return_numpy else out

    edt_input = cp.ones(cc.shape, dtype=cp.uint8)
    edt_input[cc > 0] = 0
    indices = cupy_ndimage.distance_transform_edt(
        edt_input, sampling=sampling, return_distances=False, return_indices=True
    )
    voronoi = cc[tuple(indices)]
    return cp.asnumpy(voronoi) if return_numpy else voronoi


if _has_cupy and _has_cupyx:
    class CCBaseMetricGPU(CCBaseMetric):  # type: ignore[misc]
        """
        GPU-accelerated preprocessing (Voronoi/labeling via CuPy); evaluation remains in MONAI CPU metrics.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # kwargs forwarded to base metric
            super().__init__(*args, **kwargs)
            self.xp = cp
            self.backend = "cupy"
            self._space_separation = _compute_voronoi_regions_fast_gpu

        def _verify_and_convert(self, y_pred: Any, y: Any):
            # Convert numpy -> cupy, torch -> cupy (dlpack if on CUDA)
            if isinstance(y_pred, np.ndarray):
                y_pred = cp.asarray(y_pred)
            if isinstance(y, np.ndarray):
                y = cp.asarray(y)
            if isinstance(y_pred, torch.Tensor):
                if y_pred.is_cuda:
                    y_pred = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y_pred))
                else:
                    y_pred = cp.asarray(y_pred.detach().numpy())
            if isinstance(y, torch.Tensor):
                if y.is_cuda:
                    y = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y))
                else:
                    y = cp.asarray(y.detach().numpy())

            if not isinstance(y_pred, cp.ndarray) or not isinstance(y, cp.ndarray):
                raise TypeError("y_pred and y must be cupy arrays, numpy arrays, or torch tensors")

            # shape checks
            if len(y_pred.shape) != 5 or len(y.shape) != 5:
                raise AssertionError("Expected (B,C,D,H,W)")
            if y_pred.shape != y.shape:
                raise AssertionError(f"Input shapes do not match: {y_pred.shape} vs {y.shape}")
            if y_pred.shape[1] != 2 or y.shape[1] != 2:
                raise AssertionError("Binary (2-class) inputs required")
            if y_pred.shape[0] != 1 or y.shape[0] != 1:
                raise AssertionError("Only batch size of 1 is supported")

            # ensure consistent dtype
            y_pred = y_pred.astype(cp.float64, copy=False)
            y = y.astype(cp.float64, copy=False)
            return y_pred, y

        def _convert_to_target(self, y_pred: Any, y: Any):
            # cupy -> torch CPU via DLPack
            y_pred_t = torch.from_dlpack(cp.asarray(y_pred).toDlpack()).cpu()
            y_t = torch.from_dlpack(cp.asarray(y).toDlpack()).cpu()
            return y_pred_t, y_t

    class CCDiceMetricGPU(CCBaseMetricGPU):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(DiceMetric, *args, metric_best_score=1.0, metric_worst_score=0.0, **kwargs)

    class CCHausdorffDistanceMetricGPU(CCBaseMetricGPU):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                HausdorffDistanceMetric, *args, metric_best_score=0.0, metric_worst_score=float("inf"), **kwargs
            )

    class CCHausdorffDistance95MetricGPU(CCBaseMetricGPU):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                HausdorffDistanceMetric,
                *args,
                metric_best_score=0.0,
                metric_worst_score=float("inf"),
                percentile=95,
                **kwargs,
            )

    class CCSurfaceDistanceMetricGPU(CCBaseMetricGPU):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                SurfaceDistanceMetric, *args, metric_best_score=0.0, metric_worst_score=float("inf"), **kwargs
            )

    class CCSurfaceDiceMetricGPU(CCBaseMetricGPU):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(SurfaceDiceMetric, *args, metric_best_score=1.0, metric_worst_score=0.0, **kwargs)

    __all__ += [
        "CCBaseMetricGPU",
        "CCDiceMetricGPU",
        "CCHausdorffDistanceMetricGPU",
        "CCHausdorffDistance95MetricGPU",
        "CCSurfaceDistanceMetricGPU",
        "CCSurfaceDiceMetricGPU",
    ]

