"""Peak-level precision and recall metrics for proton NMR spectra.

Peak detection is performed after independently clipping and scaling each
spectrum to [0, 1]. This keeps the absolute SciPy peak thresholds in the same
intensity frame for targets and model predictions.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


DEFAULT_PEAK_PROMINENCE_THRESHOLD = 0.03
DEFAULT_PEAK_HEIGHT_THRESHOLD = 0.01
DEFAULT_PEAK_MATCH_TOLERANCE_POINTS = 2
DEFAULT_ADJACENT_PEAK_VALLEY_FRACTION = 0.1
DEFAULT_PEAK_INTENSITY_MATCH_FRACTION = 0.5


def normalize_nmr_spectrum(signal: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Clip negative values and independently scale a spectrum to unit height."""
    clipped = np.maximum(np.asarray(signal, dtype=np.float32), 0.0)
    maximum = float(np.max(clipped))
    if maximum <= eps:
        return np.zeros_like(clipped)
    return clipped / np.float32(maximum)


def detect_nmr_peaks(
    signal: np.ndarray,
    prominence_threshold: float,
    height_threshold: float,
):
    clipped_signal = np.maximum(np.asarray(signal, dtype=np.float32), 0.0)
    peaks, properties = find_peaks(
        clipped_signal,
        prominence=float(prominence_threshold),
        height=float(height_threshold),
    )
    return peaks.astype(np.int32), properties, clipped_signal


def merge_adjacent_peaks_by_valley(
    signal: np.ndarray,
    peaks: np.ndarray,
    valley_fraction: float | None,
):
    peak_list = [int(peak) for peak in peaks]
    if valley_fraction is None or len(peak_list) <= 1:
        return peaks.astype(np.int32), [[int(peak)] for peak in peak_list]

    groups: list[list[int]] = [[peak_list[0]]]
    for left_peak, right_peak in zip(peak_list[:-1], peak_list[1:]):
        valley_value = float(np.min(signal[left_peak : right_peak + 1]))
        smaller_peak_height = min(float(signal[left_peak]), float(signal[right_peak]))
        separation_threshold = float(valley_fraction) * smaller_peak_height
        if valley_value <= separation_threshold:
            groups.append([right_peak])
        else:
            groups[-1].append(right_peak)

    merged_peaks = np.asarray(
        [max(group, key=lambda peak_idx: float(signal[peak_idx])) for group in groups],
        dtype=np.int32,
    )
    return merged_peaks, groups


def match_nmr_peaks(
    target_signal: np.ndarray,
    prediction_signal: np.ndarray,
    prominence_threshold: float = DEFAULT_PEAK_PROMINENCE_THRESHOLD,
    height_threshold: float = DEFAULT_PEAK_HEIGHT_THRESHOLD,
    tolerance_points: int = DEFAULT_PEAK_MATCH_TOLERANCE_POINTS,
    adjacent_peak_valley_fraction: float | None = DEFAULT_ADJACENT_PEAK_VALLEY_FRACTION,
    peak_intensity_match_fraction: float | None = DEFAULT_PEAK_INTENSITY_MATCH_FRACTION,
):
    """Match normalized predicted and target NMR peak groups."""
    target_peaks, target_props, target_clipped = detect_nmr_peaks(
        target_signal,
        prominence_threshold=prominence_threshold,
        height_threshold=height_threshold,
    )
    prediction_peaks, prediction_props, prediction_clipped = detect_nmr_peaks(
        prediction_signal,
        prominence_threshold=prominence_threshold,
        height_threshold=height_threshold,
    )
    raw_target_peaks = target_peaks.copy()
    raw_prediction_peaks = prediction_peaks.copy()
    target_peaks, target_peak_groups = merge_adjacent_peaks_by_valley(
        target_clipped,
        target_peaks,
        valley_fraction=adjacent_peak_valley_fraction,
    )
    prediction_peaks, prediction_peak_groups = merge_adjacent_peaks_by_valley(
        prediction_clipped,
        prediction_peaks,
        valley_fraction=adjacent_peak_valley_fraction,
    )

    def build_group_interval(group: list[int]) -> tuple[int, int]:
        start = max(min(group) - int(tolerance_points), 0)
        end = min(max(group) + int(tolerance_points), len(target_clipped) - 1)
        return start, end

    target_group_intervals = [build_group_interval(group) for group in target_peak_groups]
    prediction_group_intervals = [
        build_group_interval(group) for group in prediction_peak_groups
    ]

    def intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return not (left[1] < right[0] or right[1] < left[0])

    matched_target_indices: set[int] = set()
    underpowered_target_indices: set[int] = set()
    supported_prediction_indices: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []
    underpowered_pairs: list[tuple[int, int]] = []

    for target_idx, (target_peak, target_interval) in enumerate(
        zip(target_peaks, target_group_intervals)
    ):
        overlapping_prediction_indices = [
            prediction_idx
            for prediction_idx, prediction_interval in enumerate(
                prediction_group_intervals
            )
            if intervals_overlap(target_interval, prediction_interval)
        ]
        if not overlapping_prediction_indices:
            continue

        supported_prediction_indices.update(overlapping_prediction_indices)
        target_height = float(target_clipped[target_peak])
        prediction_region_max = float(
            np.max(prediction_clipped[target_interval[0] : target_interval[1] + 1])
        )
        best_prediction_idx = min(
            overlapping_prediction_indices,
            key=lambda prediction_idx: (
                abs(int(prediction_peaks[prediction_idx]) - int(target_peak)),
                -float(prediction_clipped[prediction_peaks[prediction_idx]]),
            ),
        )
        best_prediction_peak = int(prediction_peaks[best_prediction_idx])
        if peak_intensity_match_fraction is not None:
            required_height = float(peak_intensity_match_fraction) * target_height
            if prediction_region_max < required_height:
                underpowered_target_indices.add(target_idx)
                underpowered_pairs.append((int(target_peak), best_prediction_peak))
                continue

        matched_target_indices.add(target_idx)
        matched_pairs.append((int(target_peak), best_prediction_peak))

    unmatched_target_peaks = [
        int(target_peaks[idx])
        for idx in range(len(target_peaks))
        if idx not in matched_target_indices and idx not in underpowered_target_indices
    ]
    unmatched_prediction_peaks = [
        int(prediction_peaks[idx])
        for idx in range(len(prediction_peaks))
        if idx not in supported_prediction_indices
    ]

    true_positive_count = len(matched_target_indices)
    supported_prediction_count = len(supported_prediction_indices)
    underpowered_count = len(underpowered_pairs)
    false_negative_count = len(unmatched_target_peaks) + underpowered_count
    false_positive_count = len(unmatched_prediction_peaks)
    precision = supported_prediction_count / max(len(prediction_peaks), 1)
    recall = true_positive_count / max(len(target_peaks), 1)

    return {
        "target_peaks": target_peaks,
        "prediction_peaks": prediction_peaks,
        "raw_target_peaks": raw_target_peaks,
        "raw_prediction_peaks": raw_prediction_peaks,
        "target_peak_groups": target_peak_groups,
        "prediction_peak_groups": prediction_peak_groups,
        "target_group_intervals": target_group_intervals,
        "prediction_group_intervals": prediction_group_intervals,
        "matched_pairs": matched_pairs,
        "underpowered_pairs": underpowered_pairs,
        "unmatched_target_peaks": unmatched_target_peaks,
        "unmatched_prediction_peaks": unmatched_prediction_peaks,
        "total_target_peaks": int(len(target_peaks)),
        "true_positive_count": int(true_positive_count),
        "supported_prediction_count": int(supported_prediction_count),
        "underpowered_count": int(underpowered_count),
        "false_negative_count": int(false_negative_count),
        "false_positive_count": int(false_positive_count),
        "precision": float(precision),
        "recall": float(recall),
        "target_peak_heights": target_props.get(
            "peak_heights", np.asarray([], dtype=np.float32)
        ),
        "prediction_peak_heights": prediction_props.get(
            "peak_heights", np.asarray([], dtype=np.float32)
        ),
    }
