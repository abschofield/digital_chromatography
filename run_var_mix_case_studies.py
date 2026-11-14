"""Evaluate pure, equal binary, and ternary inputs with a variable IR model."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from Train_fp_var_mix import (
    _compute_inference_metrics,
    _decompose_prediction_components,
    instantiate_model,
    read_smiles_list,
)
from nmr_peak_metrics import (
    DEFAULT_PEAK_HEIGHT_THRESHOLD,
    DEFAULT_PEAK_INTENSITY_MATCH_FRACTION,
    DEFAULT_PEAK_MATCH_TOLERANCE_POINTS,
    DEFAULT_PEAK_PROMINENCE_THRESHOLD,
    match_nmr_peaks,
    normalize_nmr_spectrum,
)


SUMMARY_METRIC_NAMES = (
    "mse",
    "mae",
    "sdr",
    "sir",
    "sar",
    "peak_abs_error",
)

NMR_FIELDNAMES = (
    "precision",
    "recall",
    "total_target_peaks",
    "predicted_peaks",
    "true_positive_count",
    "underpowered_count",
    "false_negative_count",
    "false_positive_count",
)

CASE_DEFINITIONS = (
    {
        "case_tag": "case_01_pure_primary",
        "case_label": "Pure Spectrum (Target Absent)",
        "description": (
            "Two-molecule adjacent-cycle order with target weight 0.0 and "
            "primary weight 1.0. This matches the alpha=0 edge case on the "
            "full test set and probes whether the model outputs near-zero "
            "target signal for pure non-target spectra."
        ),
        "weights": (0.0, 1.0),
        "component_roles": ("target", "primary"),
        "pairing_mode": "adjacent_cycle_pairs",
    },
    {
        "case_tag": "case_02_equal_mix",
        "case_label": "Equal-Intensity Two-Component Mixtures",
        "description": (
            "Two-molecule adjacent-cycle order with equal weights 0.5 / 0.5."
        ),
        "weights": (0.5, 0.5),
        "component_roles": ("target", "primary"),
        "pairing_mode": "adjacent_cycle_pairs",
    },
    {
        "case_tag": "case_03_three_component",
        "case_label": "Three-Component Mixtures",
        "description": (
            "Three adjacent molecules in the shared shuffled test order with "
            "weights 0.1 / 0.25 / 0.65."
        ),
        "weights": (0.1, 0.25, 0.65),
        "component_roles": ("target", "primary", "tertiary"),
        "pairing_mode": "adjacent_cycle_triples",
    },
    {
        "case_tag": "case_04_recursive_three_component",
        "case_label": "Recursive Three-Component Mixtures",
        "description": (
            "Same adjacent three-molecule order and weights as case 3 "
            "(0.1 / 0.25 / 0.65), but also stores second-pass model outputs "
            "after feeding the scaled first-pass prediction and the scaled "
            "input residual back into the model."
        ),
        "weights": (0.1, 0.25, 0.65),
        "component_roles": ("target", "primary", "tertiary"),
        "pairing_mode": "adjacent_cycle_triples",
        "recursive_second_pass": True,
        "export_examples": False,
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full-test case studies for variable-mixture checkpoints."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--split", type=str, choices=("val", "test"), default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--num-example-exports", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--case-tags", type=str, nargs="*", default=None)
    parser.add_argument(
        "--disable-model-autocast",
        dest="disable_model_autocast",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--keep-model-autocast",
        dest="disable_model_autocast",
        action="store_false",
    )
    parser.add_argument("--nmr", dest="nmr", action="store_true", default=None)
    parser.add_argument("--no-nmr", dest="nmr", action="store_false")
    parser.add_argument(
        "--nmr-peak-prominence-threshold",
        type=float,
        default=DEFAULT_PEAK_PROMINENCE_THRESHOLD,
    )
    parser.add_argument(
        "--nmr-peak-height-threshold",
        type=float,
        default=DEFAULT_PEAK_HEIGHT_THRESHOLD,
    )
    parser.add_argument(
        "--nmr-peak-match-tolerance-points",
        type=int,
        default=DEFAULT_PEAK_MATCH_TOLERANCE_POINTS,
    )
    parser.add_argument(
        "--nmr-peak-intensity-match-fraction",
        type=float,
        default=DEFAULT_PEAK_INTENSITY_MATCH_FRACTION,
    )
    return parser.parse_args()


def _unwrap_callable(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _sample_without_model_autocast(model, mix, num_steps):
    sample_fn = _unwrap_callable(type(model).sample)
    autocast_context = (
        torch.autocast(device_type="cuda", enabled=False)
        if mix.is_cuda
        else nullcontext()
    )
    with torch.no_grad():
        with autocast_context:
            pred, returned_mix = sample_fn(
                model,
                mix.float(),
                num_steps=num_steps,
                show_progress=False,
            )
    return pred, returned_mix


def _normalize_batch_zero_one(tensor: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, tensor.ndim))
    value_min = tensor.amin(dim=dims, keepdim=True)
    value_max = tensor.amax(dim=dims, keepdim=True)
    denom = (value_max - value_min).clamp_min(1e-8)
    return (tensor - value_min) / denom


def _normalize_batch_by_max(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dims = tuple(range(1, tensor.ndim))
    scale = tensor.amax(dim=dims, keepdim=True).clamp_min(1e-8)
    return tensor / scale, scale


def _scale_nonnegative_batch_to_one(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scaled, scale = _normalize_batch_by_max(tensor.clamp_min(0.0))
    return scaled, scale.view(tensor.shape[0])


def _set_sampling_seed(seed: int):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _suppress_bss_metrics_for_zero_target(metrics: dict[str, torch.Tensor], zero_target_mask: torch.Tensor):
    if zero_target_mask is None or zero_target_mask.numel() == 0:
        return metrics
    if not torch.any(zero_target_mask):
        return metrics
    updated = dict(metrics)
    for key in ("infer_sdr", "infer_sir", "infer_sar"):
        tensor = updated[key].clone()
        tensor[zero_target_mask] = torch.nan
        updated[key] = tensor
    return updated


def _load_dataset_split(cfg, split):
    data_cfg = cfg.dataset
    df = pd.read_csv(
        data_cfg.data_csv,
        index_col=0,
        dtype=np.float32,
        converters={0: str},
    )
    split_smiles = {
        "val": read_smiles_list(data_cfg.val_smiles),
        "test": read_smiles_list(data_cfg.test_smiles),
    }[split]
    split_spectra = torch.from_numpy(df.loc[split_smiles].to_numpy(dtype=np.float32))
    return split_spectra, split_smiles


def _build_adjacent_cycle_order(num_items: int, selection_seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(int(selection_seed))
    return torch.randperm(num_items, generator=generator).tolist()


def _build_case_records(order: list[int], split_smiles: list[str], case_def: dict):
    num_items = len(order)
    records = []
    roles = case_def["component_roles"]
    weights = case_def["weights"]
    for sample_index in range(num_items):
        component_indices = [order[(sample_index + offset) % num_items] for offset in range(len(roles))]
        role_map = {role: component_indices[idx] for idx, role in enumerate(roles)}
        records.append(
            {
                "sample_index": sample_index,
                "target_index": role_map["target"],
                "primary_index": role_map.get("primary", -1),
                "tertiary_index": role_map.get("tertiary", -1),
                "target_smiles": split_smiles[role_map["target"]],
                "primary_smiles": split_smiles[role_map["primary"]] if "primary" in role_map else "",
                "tertiary_smiles": split_smiles[role_map["tertiary"]] if "tertiary" in role_map else "",
                "component_indices": component_indices,
                "component_roles": roles,
                "component_weights": weights,
                "interference_smiles": " | ".join(
                    split_smiles[component_indices[idx]]
                    for idx in range(1, len(component_indices))
                ),
            }
        )
    return records


def _build_case_batch(split_spectra: torch.Tensor, batch_records, case_def: dict):
    roles = list(case_def["component_roles"])
    weights = [float(value) for value in case_def["weights"]]
    num_samples = len(batch_records)

    pure_components = {}
    weighted_components = {}
    scaled_components = {}
    raw_components = {}
    for role_idx, role in enumerate(roles):
        indices = torch.tensor(
            [record["component_indices"][role_idx] for record in batch_records],
            dtype=torch.long,
        )
        raw = split_spectra.index_select(0, indices).unsqueeze(1)
        pure = _normalize_batch_zero_one(raw)
        weighted = pure * weights[role_idx]
        raw_components[role] = raw
        pure_components[role] = pure
        weighted_components[role] = weighted

    mixture_pre = sum(weighted_components.values())
    mixture, mixture_scale = _normalize_batch_by_max(mixture_pre)
    for role in roles:
        scaled_components[role] = weighted_components[role] / mixture_scale

    target_reference = scaled_components["target"]
    combined_interference = sum(
        scaled_components[role] for role in roles if role != "target"
    )
    zero_target_mask = torch.tensor(
        [float(record["component_weights"][0]) == 0.0 for record in batch_records],
        dtype=torch.bool,
    )
    target_max_intensity = target_reference.amax(dim=(1, 2))
    target_display_max = target_max_intensity.clone()
    target_display_max[zero_target_mask] = 0.1

    return {
        "roles": roles,
        "weights": weights,
        "mixture": mixture,
        "mixture_pre": mixture_pre,
        "mixture_scale": mixture_scale.view(num_samples),
        "target_reference": target_reference,
        "interference_reference": combined_interference,
        "pure_components": pure_components,
        "weighted_components": weighted_components,
        "scaled_components": scaled_components,
        "target_max_intensity": target_max_intensity,
        "target_display_max": target_display_max,
        "zero_target_mask": zero_target_mask,
    }


def _metric_arrays(metrics: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "mse": metrics["infer_mse"].detach().cpu().numpy().astype(np.float64),
        "mae": metrics["infer_mae"].detach().cpu().numpy().astype(np.float64),
        "sdr": metrics["infer_sdr"].detach().cpu().numpy().astype(np.float64),
        "sir": metrics["infer_sir"].detach().cpu().numpy().astype(np.float64),
        "sar": metrics["infer_sar"].detach().cpu().numpy().astype(np.float64),
        "peak_abs_error": metrics["infer_peak_abs_error"].detach().cpu().numpy().astype(np.float64),
    }


def _compute_nmr_peak_metric_rows(
    pred_batch: torch.Tensor,
    target_batch: torch.Tensor,
    zero_target_mask: torch.Tensor,
    prominence_threshold: float,
    height_threshold: float,
    tolerance_points: int,
    peak_intensity_match_fraction: float | None,
):
    pred_np = pred_batch.detach().cpu().numpy()
    target_np = target_batch.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[:, 0, :]
    if target_np.ndim == 3:
        target_np = target_np[:, 0, :]
    zero_mask_np = zero_target_mask.detach().cpu().numpy().astype(bool)

    rows = []
    for pred_signal, target_signal, zero_target in zip(pred_np, target_np, zero_mask_np):
        if zero_target:
            rows.append(
                {
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "total_target_peaks": 0,
                    "predicted_peaks": 0,
                    "true_positive_count": 0,
                    "underpowered_count": 0,
                    "false_negative_count": 0,
                    "false_positive_count": 0,
                }
            )
            continue
        peak_match = match_nmr_peaks(
            target_signal=normalize_nmr_spectrum(target_signal),
            prediction_signal=normalize_nmr_spectrum(pred_signal),
            prominence_threshold=prominence_threshold,
            height_threshold=height_threshold,
            tolerance_points=tolerance_points,
            adjacent_peak_valley_fraction=None,
            peak_intensity_match_fraction=peak_intensity_match_fraction,
        )
        rows.append(
            {
                "precision": float(peak_match["precision"]),
                "recall": float(peak_match["recall"]),
                "total_target_peaks": int(peak_match["total_target_peaks"]),
                "predicted_peaks": int(len(peak_match["prediction_peaks"])),
                "true_positive_count": int(peak_match["true_positive_count"]),
                "underpowered_count": int(peak_match["underpowered_count"]),
                "false_negative_count": int(peak_match["false_negative_count"]),
                "false_positive_count": int(peak_match["false_positive_count"]),
            }
        )
    return rows


def _append_summary_block(summary_acc, condition: str, metrics: dict[str, torch.Tensor], nmr_rows=None):
    block = summary_acc.setdefault(
        condition,
        {name: [] for name in SUMMARY_METRIC_NAMES}
        | {name: [] for name in NMR_FIELDNAMES},
    )
    arrays = _metric_arrays(metrics)
    for metric_name in SUMMARY_METRIC_NAMES:
        block[metric_name].extend(arrays[metric_name].tolist())
    if nmr_rows is not None:
        for row in nmr_rows:
            for name in NMR_FIELDNAMES:
                block[name].append(float(row[name]))


def _write_summary_csv(path: Path, summary_acc, case_def: dict, num_steps: int):
    fieldnames = ["case_tag", "case_label", "condition", "steps", "count"]
    for metric_name in SUMMARY_METRIC_NAMES:
        fieldnames.extend(
            [
                f"mean_{metric_name}",
                f"median_{metric_name}",
                f"std_{metric_name}",
                f"finite_{metric_name}_fraction",
            ]
        )
    for name in NMR_FIELDNAMES:
        fieldnames.append(f"mean_{name}")

    rows = []
    for condition, values in summary_acc.items():
        row = {
            "case_tag": case_def["case_tag"],
            "case_label": case_def["case_label"],
            "condition": condition,
            "steps": int(num_steps),
            "count": len(values["mse"]),
        }
        for metric_name in SUMMARY_METRIC_NAMES:
            arr = np.asarray(values[metric_name], dtype=np.float64)
            finite = np.isfinite(arr)
            finite_values = arr[finite]
            row[f"mean_{metric_name}"] = float(np.mean(finite_values)) if finite_values.size else ""
            row[f"median_{metric_name}"] = float(np.median(finite_values)) if finite_values.size else ""
            row[f"std_{metric_name}"] = float(np.std(finite_values)) if finite_values.size else ""
            row[f"finite_{metric_name}_fraction"] = float(np.mean(finite)) if arr.size else ""
        for name in NMR_FIELDNAMES:
            arr = np.asarray(values[name], dtype=np.float64)
            finite = np.isfinite(arr)
            finite_values = arr[finite]
            row[f"mean_{name}"] = float(np.mean(finite_values)) if finite_values.size else ""
        rows.append(row)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_examples_zero_one(tensor: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, tensor.ndim))
    value_min = tensor.amin(dim=dims, keepdim=True)
    value_max = tensor.amax(dim=dims, keepdim=True)
    denom = (value_max - value_min).clamp_min(1e-8)
    return (tensor - value_min) / denom


def _prepare_case_dir(case_dir: Path):
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "full_outputs_chunks").mkdir(parents=True, exist_ok=True)
    example_dirs = list(case_dir.glob("eval_examples_*"))
    for example_dir in example_dirs:
        for item in example_dir.glob("*"):
            if item.is_file():
                item.unlink()
        example_dir.rmdir()
    for chunk_file in (case_dir / "full_outputs_chunks").glob("chunk_*.npz"):
        chunk_file.unlink()
    for filename in (
        "manifest.json",
        "raw_metrics.csv",
        "summary_metrics.csv",
        "sample_manifest.csv",
    ):
        path = case_dir / filename
        if path.exists():
            path.unlink()


def _merge_case_manifest(output_root: Path, top_manifest: dict):
    manifest_path = output_root / "case_studies_manifest.json"
    existing_cases = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                existing_manifest = json.load(handle)
            for case_manifest in existing_manifest.get("cases", []):
                case_tag = case_manifest.get("case_tag")
                if case_tag:
                    existing_cases[case_tag] = case_manifest
        except Exception:
            existing_cases = {}
    for case_manifest in top_manifest["cases"]:
        existing_cases[case_manifest["case_tag"]] = case_manifest
    top_manifest["cases"] = [existing_cases[key] for key in sorted(existing_cases)]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(top_manifest, handle, indent=2)


def _write_full_output_chunk(case_dir: Path, batch_idx: int, batch_records, batch, pred_cpu):
    roles = batch["roles"]
    chunk_path = case_dir / "full_outputs_chunks" / f"chunk_{batch_idx:05d}.npz"
    target_index = np.asarray([record["target_index"] for record in batch_records], dtype=np.int32)
    primary_index = np.asarray([record["primary_index"] for record in batch_records], dtype=np.int32)
    tertiary_index = np.asarray([record["tertiary_index"] for record in batch_records], dtype=np.int32)
    sample_index = np.asarray([record["sample_index"] for record in batch_records], dtype=np.int32)
    arrays = {
        "sample_index": sample_index,
        "target_index": target_index,
        "primary_index": primary_index,
        "tertiary_index": tertiary_index,
        "mixture_spectra": batch["mixture"].numpy().astype(np.float32)[:, 0, :],
        "prediction_spectra": pred_cpu.numpy().astype(np.float32)[:, 0, :],
        "target_reference_spectra": batch["target_reference"].numpy().astype(np.float32)[:, 0, :],
        "interference_reference_spectra": batch["interference_reference"].numpy().astype(np.float32)[:, 0, :],
        "target_weighted_component": batch["scaled_components"]["target"].numpy().astype(np.float32)[:, 0, :],
        "interference_weighted_component": batch["interference_reference"].numpy().astype(np.float32)[:, 0, :],
    }
    if "primary" in roles:
        arrays["primary_weighted_component"] = batch["scaled_components"]["primary"].numpy().astype(np.float32)[:, 0, :]
    if "tertiary" in roles:
        arrays["tertiary_weighted_component"] = batch["scaled_components"]["tertiary"].numpy().astype(np.float32)[:, 0, :]
    np.savez_compressed(chunk_path, **arrays)
    return chunk_path


def _write_recursive_case04_full_output_chunk(
    case_dir: Path,
    batch_idx: int,
    batch_records,
    batch,
    pred_cpu,
    initial_residual_cpu,
    scaled_prediction_input_cpu,
    scaled_residual_input_cpu,
    prediction_prediction_cpu,
    residual_prediction_cpu,
    prediction_input_scale,
    residual_input_scale,
):
    chunk_path = case_dir / "full_outputs_chunks" / f"chunk_{batch_idx:05d}.npz"
    target_index = np.asarray([record["target_index"] for record in batch_records], dtype=np.int32)
    primary_index = np.asarray([record["primary_index"] for record in batch_records], dtype=np.int32)
    tertiary_index = np.asarray([record["tertiary_index"] for record in batch_records], dtype=np.int32)
    sample_index = np.asarray([record["sample_index"] for record in batch_records], dtype=np.int32)
    arrays = {
        "sample_index": sample_index,
        "target_index": target_index,
        "primary_index": primary_index,
        "tertiary_index": tertiary_index,
        "input_spectra": batch["mixture"].numpy().astype(np.float32)[:, 0, :],
        "initial_prediction_spectra": pred_cpu.numpy().astype(np.float32)[:, 0, :],
        "initial_residual_spectra": initial_residual_cpu.numpy().astype(np.float32)[:, 0, :],
        "scaled_prediction_input_spectra": scaled_prediction_input_cpu.numpy().astype(np.float32)[:, 0, :],
        "scaled_residual_input_spectra": scaled_residual_input_cpu.numpy().astype(np.float32)[:, 0, :],
        "prediction_prediction_spectra": prediction_prediction_cpu.numpy().astype(np.float32)[:, 0, :],
        "residual_prediction_spectra": residual_prediction_cpu.numpy().astype(np.float32)[:, 0, :],
        "prediction_input_scale": prediction_input_scale.numpy().astype(np.float32),
        "residual_input_scale": residual_input_scale.numpy().astype(np.float32),
        "pure_minority_spectra": batch["pure_components"]["target"].numpy().astype(np.float32)[:, 0, :],
        "pure_intermediate_spectra": batch["pure_components"]["primary"].numpy().astype(np.float32)[:, 0, :],
        "pure_primary_spectra": batch["pure_components"]["tertiary"].numpy().astype(np.float32)[:, 0, :],
        "minority_reference_spectra": batch["target_reference"].numpy().astype(np.float32)[:, 0, :],
        "combined_interference_spectra": batch["interference_reference"].numpy().astype(np.float32)[:, 0, :],
    }
    np.savez_compressed(chunk_path, **arrays)
    return chunk_path


def _write_example_export(
    example_dir: Path,
    case_def: dict,
    cfg,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    split: str,
    num_steps: int,
    disable_model_autocast: bool,
    selection_seed: int,
    nmr_enabled: bool,
    nmr_settings: dict,
    example_rows,
):
    example_dir.mkdir(parents=True, exist_ok=True)
    for path in example_dir.glob("example_*.npz"):
        path.unlink()
    manifest = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "split_used": split,
        "num_steps": int(num_steps),
        "disable_model_autocast": bool(disable_model_autocast),
        "num_examples": len(example_rows),
        "selection_seed": int(selection_seed),
        "case_tag": case_def["case_tag"],
        "case_label": case_def["case_label"],
        "case_description": case_def["description"],
        "nmr_enabled": bool(nmr_enabled),
        "nmr_settings": nmr_settings,
        "examples": [],
    }

    summary_fieldnames = [
        "example_index",
        "sample_index",
        "target_index",
        "interference_index",
        "primary_index",
        "tertiary_index",
        "target_smiles",
        "interference_smiles",
        "primary_smiles",
        "tertiary_smiles",
        "target_weight",
        "primary_weight",
        "tertiary_weight",
        "mse",
        "mae",
        "sdr",
        "sir",
        "sar",
        "peak_abs_error",
        "input_target_sdr",
        "input_target_sir",
        "input_target_sar",
        "input_primary_sdr",
        "input_primary_sir",
        "input_primary_sar",
        "residual_primary_sdr",
        "residual_primary_sir",
        "residual_primary_sar",
        "component_sum_max_abs_error",
    ]
    if nmr_enabled:
        summary_fieldnames.extend(
            [
                "precision",
                "recall",
                "total_target_peaks",
                "predicted_peaks",
                "true_positive_count",
                "underpowered_count",
                "false_negative_count",
                "false_positive_count",
            ]
        )

    with open(example_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()

        for example_index, row in enumerate(example_rows):
            example_path = example_dir / f"example_{example_index:02d}.npz"
            np.savez_compressed(
                example_path,
                point_index=np.arange(int(cfg.dataset.spec_len), dtype=np.int32),
                requested_minority_alpha=np.asarray([row["target_weight"]], dtype=np.float32),
                mixture_spectrum=row["mixture_spectrum"].astype(np.float32),
                mixture_pre_normalization=row["mixture_pre_normalization"].astype(np.float32),
                target_reference_spectrum=row["target_reference_spectrum"].astype(np.float32),
                interference_reference_spectrum=row["interference_reference_spectrum"].astype(np.float32),
                target_weighted_component=row["target_weighted_component"].astype(np.float32),
                interference_weighted_component=row["interference_weighted_component"].astype(np.float32),
                prediction_spectrum=row["prediction_spectrum"].astype(np.float32),
                signal_component=row["signal_component"].astype(np.float32),
                interference_component=row["prediction_interference_component"].astype(np.float32),
                artifact_component=row["artifact_component"].astype(np.float32),
                prediction_component_sum=row["prediction_component_sum"].astype(np.float32),
                target_unscaled_pure_spectrum=row["target_unscaled_pure_spectrum"].astype(np.float32),
                interference_unscaled_pure_spectrum=row["interference_unscaled_pure_spectrum"].astype(np.float32),
                target_display_max=np.asarray([row["target_display_max"]], dtype=np.float32),
                primary_weighted_component=row["primary_weighted_component"].astype(np.float32),
                tertiary_weighted_component=row["tertiary_weighted_component"].astype(np.float32),
            )

            manifest_row = {
                "example_index": example_index,
                "sample_index": row["sample_index"],
                "target_index": row["target_index"],
                "interference_index": row["primary_index"],
                "primary_index": row["primary_index"],
                "tertiary_index": row["tertiary_index"],
                "target_smiles": row["target_smiles"],
                "interference_smiles": row["interference_smiles"],
                "primary_smiles": row["primary_smiles"],
                "tertiary_smiles": row["tertiary_smiles"],
                "target_weight": row["target_weight"],
                "primary_weight": row["primary_weight"],
                "tertiary_weight": row["tertiary_weight"],
                "mse": row["mse"],
                "mae": row["mae"],
                "sdr": row["sdr"],
                "sir": row["sir"],
                "sar": row["sar"],
                "peak_abs_error": row["peak_abs_error"],
                "input_target_sdr": row["input_target_sdr"],
                "input_target_sir": row["input_target_sir"],
                "input_target_sar": row["input_target_sar"],
                "input_primary_sdr": row["input_primary_sdr"],
                "input_primary_sir": row["input_primary_sir"],
                "input_primary_sar": row["input_primary_sar"],
                "residual_primary_sdr": row["residual_primary_sdr"],
                "residual_primary_sir": row["residual_primary_sir"],
                "residual_primary_sar": row["residual_primary_sar"],
                "component_sum_max_abs_error": row["component_sum_max_abs_error"],
            }
            if nmr_enabled:
                manifest_row.update(
                    {
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "total_target_peaks": row["total_target_peaks"],
                        "predicted_peaks": row["predicted_peaks"],
                        "true_positive_count": row["true_positive_count"],
                        "underpowered_count": row["underpowered_count"],
                        "false_negative_count": row["false_negative_count"],
                        "false_positive_count": row["false_positive_count"],
                    }
                )
            manifest["examples"].append(manifest_row)
            writer.writerow(manifest_row)

    with open(example_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_case_studies(args):
    cfg = OmegaConf.load(args.config)
    case_cfg = cfg.get("case_studies", {})
    args.checkpoint = args.checkpoint or case_cfg.get("checkpoint_path", None)
    args.output_root = args.output_root or case_cfg.get("output_dir", None)
    args.split = args.split or case_cfg.get("split", "test")
    args.num_steps = int(
        args.num_steps if args.num_steps is not None else case_cfg.get("steps", 10)
    )
    if args.device is None:
        args.device = case_cfg.get("device", None)
    if args.nmr is None:
        args.nmr = bool(case_cfg.get("nmr", False))
    if args.disable_model_autocast is None:
        args.disable_model_autocast = bool(
            case_cfg.get(
                "disable_model_autocast",
                cfg.train.get("disable_model_autocast", False),
            )
        )
    if args.checkpoint is None:
        raise ValueError("Set case_studies.checkpoint_path in the config.")
    if args.output_root is None:
        raise ValueError("Set case_studies.output_dir in the config.")
    args.checkpoint = Path(args.checkpoint)
    args.output_root = Path(args.output_root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.batch_size)
    selection_seed = int(args.selection_seed if args.selection_seed is not None else cfg.seed)
    sampling_seed = int(args.sampling_seed if args.sampling_seed is not None else cfg.seed)
    disable_model_autocast = bool(args.disable_model_autocast)

    print(f"[case-studies] config={args.config}", flush=True)
    print(f"[case-studies] checkpoint={args.checkpoint}", flush=True)
    print(f"[case-studies] output_root={args.output_root}", flush=True)
    print(f"[case-studies] split={args.split}", flush=True)
    print(f"[case-studies] num_steps={args.num_steps}", flush=True)
    print(f"[case-studies] device={device}", flush=True)
    print(f"[case-studies] batch_size={batch_size}", flush=True)
    print(f"[case-studies] selection_seed={selection_seed}", flush=True)
    print(f"[case-studies] sampling_seed={sampling_seed}", flush=True)
    print(f"[case-studies] nmr={args.nmr}", flush=True)

    model = instantiate_model(cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device)
    model.eval()

    split_spectra, split_smiles = _load_dataset_split(cfg, args.split)
    order = _build_adjacent_cycle_order(split_spectra.shape[0], selection_seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    top_manifest = {
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "split": args.split,
        "num_steps": int(args.num_steps),
        "selection_seed": selection_seed,
        "sampling_seed": sampling_seed,
        "nmr_enabled": bool(args.nmr),
        "cases": [],
    }

    selected_case_tags = set(args.case_tags) if args.case_tags else None

    for case_def in CASE_DEFINITIONS:
        if selected_case_tags is not None and case_def["case_tag"] not in selected_case_tags:
            continue
        case_dir = args.output_root / case_def["case_tag"]
        _prepare_case_dir(case_dir)
        export_examples = bool(case_def.get("export_examples", True))
        recursive_second_pass = bool(case_def.get("recursive_second_pass", False))
        records = _build_case_records(order, split_smiles, case_def)
        if args.max_samples is not None:
            records = records[: int(args.max_samples)]
        example_rows = []
        summary_acc = {}
        _set_sampling_seed(sampling_seed)

        sample_manifest_fieldnames = [
            "sample_index",
            "chunk_file",
            "chunk_offset",
            "target_index",
            "primary_index",
            "tertiary_index",
            "minority_index",
            "intermediate_index",
            "dominant_primary_index",
            "target_smiles",
            "primary_smiles",
            "tertiary_smiles",
            "minority_smiles",
            "intermediate_smiles",
            "dominant_primary_smiles",
            "interference_smiles",
            "target_weight",
            "primary_weight",
            "tertiary_weight",
        ]
        raw_metric_fieldnames = [
            "sample_index",
            "case_tag",
            "condition",
            "steps",
            "target_index",
            "primary_index",
            "tertiary_index",
            "target_smiles",
            "primary_smiles",
            "tertiary_smiles",
            "interference_smiles",
            "target_weight",
            "primary_weight",
            "tertiary_weight",
            "target_max_intensity",
            "mixture_scale",
            "mse",
            "mae",
            "sdr",
            "sir",
            "sar",
            "peak_abs_error",
        ]
        if args.nmr:
            raw_metric_fieldnames.extend(NMR_FIELDNAMES)

        print(f"[case-studies] starting {case_def['case_tag']}", flush=True)
        with open(case_dir / "sample_manifest.csv", "w", encoding="utf-8", newline="") as sample_handle, open(
            case_dir / "raw_metrics.csv", "w", encoding="utf-8", newline=""
        ) as raw_handle:
            sample_writer = csv.DictWriter(sample_handle, fieldnames=sample_manifest_fieldnames)
            raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_metric_fieldnames)
            sample_writer.writeheader()
            raw_writer.writeheader()

            total_batches = (len(records) + batch_size - 1) // batch_size
            for batch_idx, start in enumerate(range(0, len(records), batch_size), start=1):
                end = min(len(records), start + batch_size)
                batch_records = records[start:end]
                batch = _build_case_batch(split_spectra, batch_records, case_def)

                mix_cpu = batch["mixture"].float()
                target_cpu = batch["target_reference"].float()
                interference_cpu = batch["interference_reference"].float()
                pure_target_cpu = batch["pure_components"]["target"].float()

                input_target_metrics = _compute_inference_metrics(
                    mix_cpu,
                    target_cpu,
                    interference_cpu,
                    pure_target=pure_target_cpu,
                )
                input_target_metrics = _suppress_bss_metrics_for_zero_target(
                    input_target_metrics,
                    batch["zero_target_mask"],
                )
                input_primary_metrics = _compute_inference_metrics(
                    mix_cpu,
                    interference_cpu,
                    target_cpu,
                    pure_target=interference_cpu,
                )

                mix_device = mix_cpu.to(device)
                if disable_model_autocast:
                    pred, _ = _sample_without_model_autocast(model, mix_device, num_steps=args.num_steps)
                else:
                    with torch.no_grad():
                        pred, _ = model.sample(mix_device, num_steps=args.num_steps, show_progress=False)
                pred_cpu = pred.unsqueeze(1).detach().cpu().float()

                if recursive_second_pass:
                    initial_residual_cpu = mix_cpu - pred_cpu
                    scaled_prediction_input_cpu, prediction_input_scale = _scale_nonnegative_batch_to_one(pred_cpu)
                    scaled_residual_input_cpu, residual_input_scale = _scale_nonnegative_batch_to_one(initial_residual_cpu)
                    scaled_prediction_input_device = scaled_prediction_input_cpu.to(device)
                    scaled_residual_input_device = scaled_residual_input_cpu.to(device)
                    if disable_model_autocast:
                        prediction_prediction, _ = _sample_without_model_autocast(
                            model,
                            scaled_prediction_input_device,
                            num_steps=args.num_steps,
                        )
                        residual_prediction, _ = _sample_without_model_autocast(
                            model,
                            scaled_residual_input_device,
                            num_steps=args.num_steps,
                        )
                    else:
                        with torch.no_grad():
                            prediction_prediction, _ = model.sample(
                                scaled_prediction_input_device,
                                num_steps=args.num_steps,
                                show_progress=False,
                            )
                            residual_prediction, _ = model.sample(
                                scaled_residual_input_device,
                                num_steps=args.num_steps,
                                show_progress=False,
                            )
                    prediction_prediction_cpu = prediction_prediction.unsqueeze(1).detach().cpu().float()
                    residual_prediction_cpu = residual_prediction.unsqueeze(1).detach().cpu().float()
                else:
                    initial_residual_cpu = None
                    scaled_prediction_input_cpu = None
                    scaled_residual_input_cpu = None
                    prediction_prediction_cpu = None
                    residual_prediction_cpu = None
                    prediction_input_scale = None
                    residual_input_scale = None

                prediction_metrics = _compute_inference_metrics(
                    pred_cpu,
                    target_cpu,
                    interference_cpu,
                    pure_target=pure_target_cpu,
                )
                prediction_metrics = _suppress_bss_metrics_for_zero_target(
                    prediction_metrics,
                    batch["zero_target_mask"],
                )

                if args.nmr:
                    pred_nmr_rows = _compute_nmr_peak_metric_rows(
                        pred_cpu,
                        target_cpu,
                        batch["zero_target_mask"],
                        args.nmr_peak_prominence_threshold,
                        args.nmr_peak_height_threshold,
                        args.nmr_peak_match_tolerance_points,
                        args.nmr_peak_intensity_match_fraction,
                    )
                    input_nmr_rows = _compute_nmr_peak_metric_rows(
                        mix_cpu,
                        target_cpu,
                        batch["zero_target_mask"],
                        args.nmr_peak_prominence_threshold,
                        args.nmr_peak_height_threshold,
                        args.nmr_peak_match_tolerance_points,
                        args.nmr_peak_intensity_match_fraction,
                    )
                else:
                    pred_nmr_rows = None
                    input_nmr_rows = None

                _append_summary_block(summary_acc, "prediction_target", prediction_metrics, pred_nmr_rows)
                _append_summary_block(summary_acc, "input_target_reference", input_target_metrics, input_nmr_rows)
                _append_summary_block(summary_acc, "input_primary_reference", input_primary_metrics, None)

                if recursive_second_pass:
                    chunk_path = _write_recursive_case04_full_output_chunk(
                        case_dir,
                        batch_idx,
                        batch_records,
                        batch,
                        pred_cpu,
                        initial_residual_cpu,
                        scaled_prediction_input_cpu,
                        scaled_residual_input_cpu,
                        prediction_prediction_cpu,
                        residual_prediction_cpu,
                        prediction_input_scale,
                        residual_input_scale,
                    )
                else:
                    chunk_path = _write_full_output_chunk(case_dir, batch_idx, batch_records, batch, pred_cpu)

                metric_arrays_prediction = _metric_arrays(prediction_metrics)
                metric_arrays_input_target = _metric_arrays(input_target_metrics)
                metric_arrays_input_primary = _metric_arrays(input_primary_metrics)
                target_max_np = batch["target_max_intensity"].numpy().astype(np.float64)
                mixture_scale_np = batch["mixture_scale"].numpy().astype(np.float64)

                for local_idx, record in enumerate(batch_records):
                    sample_writer.writerow(
                        {
                            "sample_index": record["sample_index"],
                            "chunk_file": str(chunk_path.relative_to(case_dir)),
                            "chunk_offset": local_idx,
                            "target_index": record["target_index"],
                            "primary_index": record["primary_index"],
                            "tertiary_index": record["tertiary_index"],
                            "minority_index": record["target_index"],
                            "intermediate_index": record["primary_index"],
                            "dominant_primary_index": record["tertiary_index"],
                            "target_smiles": record["target_smiles"],
                            "primary_smiles": record["primary_smiles"],
                            "tertiary_smiles": record["tertiary_smiles"],
                            "minority_smiles": record["target_smiles"],
                            "intermediate_smiles": record["primary_smiles"],
                            "dominant_primary_smiles": record["tertiary_smiles"],
                            "interference_smiles": record["interference_smiles"],
                            "target_weight": record["component_weights"][0],
                            "primary_weight": record["component_weights"][1] if len(record["component_weights"]) > 1 else 0.0,
                            "tertiary_weight": record["component_weights"][2] if len(record["component_weights"]) > 2 else 0.0,
                        }
                    )

                    for condition, metric_arrays, nmr_rows in (
                        ("prediction_target", metric_arrays_prediction, pred_nmr_rows),
                        ("input_target_reference", metric_arrays_input_target, input_nmr_rows),
                        ("input_primary_reference", metric_arrays_input_primary, None),
                    ):
                        raw_row = {
                            "sample_index": record["sample_index"],
                            "case_tag": case_def["case_tag"],
                            "condition": condition,
                            "steps": args.num_steps,
                            "target_index": record["target_index"],
                            "primary_index": record["primary_index"],
                            "tertiary_index": record["tertiary_index"],
                            "target_smiles": record["target_smiles"],
                            "primary_smiles": record["primary_smiles"],
                            "tertiary_smiles": record["tertiary_smiles"],
                            "interference_smiles": record["interference_smiles"],
                            "target_weight": record["component_weights"][0],
                            "primary_weight": record["component_weights"][1] if len(record["component_weights"]) > 1 else 0.0,
                            "tertiary_weight": record["component_weights"][2] if len(record["component_weights"]) > 2 else 0.0,
                            "target_max_intensity": float(target_max_np[local_idx]),
                            "mixture_scale": float(mixture_scale_np[local_idx]),
                        }
                        for metric_name in SUMMARY_METRIC_NAMES:
                            raw_row[metric_name] = float(metric_arrays[metric_name][local_idx])
                        if args.nmr:
                            if nmr_rows is not None:
                                raw_row.update(nmr_rows[local_idx])
                            else:
                                for field in NMR_FIELDNAMES:
                                    raw_row[field] = ""
                        raw_writer.writerow(raw_row)

                    if export_examples and len(example_rows) < args.num_example_exports:
                        single_pred = pred_cpu[local_idx : local_idx + 1]
                        single_mix = mix_cpu[local_idx : local_idx + 1]
                        single_target = target_cpu[local_idx : local_idx + 1]
                        single_interference = interference_cpu[local_idx : local_idx + 1]

                        components = _decompose_prediction_components(
                            single_pred,
                            single_target,
                            single_interference,
                        )
                        primary_residual = _normalize_examples_zero_one(single_mix - single_pred)
                        residual_metrics = _compute_inference_metrics(
                            primary_residual,
                            single_interference,
                            single_target,
                            pure_target=single_interference,
                        )
                        example_row = {
                            "sample_index": record["sample_index"],
                            "target_index": record["target_index"],
                            "primary_index": record["primary_index"],
                            "tertiary_index": record["tertiary_index"],
                            "target_smiles": record["target_smiles"],
                            "primary_smiles": record["primary_smiles"],
                            "tertiary_smiles": record["tertiary_smiles"],
                            "interference_smiles": record["interference_smiles"],
                            "target_weight": record["component_weights"][0],
                            "primary_weight": record["component_weights"][1] if len(record["component_weights"]) > 1 else 0.0,
                            "tertiary_weight": record["component_weights"][2] if len(record["component_weights"]) > 2 else 0.0,
                            "mixture_spectrum": single_mix[0, 0].numpy(),
                            "mixture_pre_normalization": batch["mixture_pre"][local_idx, 0].numpy(),
                            "target_reference_spectrum": single_target[0, 0].numpy(),
                            "interference_reference_spectrum": single_interference[0, 0].numpy(),
                            "target_weighted_component": batch["scaled_components"]["target"][local_idx, 0].numpy(),
                            "interference_weighted_component": batch["interference_reference"][local_idx, 0].numpy(),
                            "primary_weighted_component": (
                                batch["scaled_components"]["primary"][local_idx, 0].numpy()
                                if "primary" in batch["scaled_components"]
                                else np.zeros_like(single_mix[0, 0].numpy())
                            ),
                            "tertiary_weighted_component": (
                                batch["scaled_components"]["tertiary"][local_idx, 0].numpy()
                                if "tertiary" in batch["scaled_components"]
                                else np.zeros_like(single_mix[0, 0].numpy())
                            ),
                            "prediction_spectrum": single_pred[0, 0].numpy(),
                            "signal_component": components["signal_component"][0, 0].numpy(),
                            "prediction_interference_component": components["interference_component"][0, 0].numpy(),
                            "artifact_component": components["artifact_component"][0, 0].numpy(),
                            "prediction_component_sum": (
                                components["signal_component"][0, 0]
                                + components["interference_component"][0, 0]
                                + components["artifact_component"][0, 0]
                            ).numpy(),
                            "target_unscaled_pure_spectrum": batch["pure_components"]["target"][local_idx, 0].numpy(),
                            "interference_unscaled_pure_spectrum": batch["interference_reference"][local_idx, 0].numpy(),
                            "target_display_max": float(batch["target_display_max"][local_idx].item()),
                            "mse": float(metric_arrays_prediction["mse"][local_idx]),
                            "mae": float(metric_arrays_prediction["mae"][local_idx]),
                            "sdr": float(metric_arrays_prediction["sdr"][local_idx]),
                            "sir": float(metric_arrays_prediction["sir"][local_idx]),
                            "sar": float(metric_arrays_prediction["sar"][local_idx]),
                            "peak_abs_error": float(metric_arrays_prediction["peak_abs_error"][local_idx]),
                            "input_target_sdr": float(metric_arrays_input_target["sdr"][local_idx]),
                            "input_target_sir": float(metric_arrays_input_target["sir"][local_idx]),
                            "input_target_sar": float(metric_arrays_input_target["sar"][local_idx]),
                            "input_primary_sdr": float(metric_arrays_input_primary["sdr"][local_idx]),
                            "input_primary_sir": float(metric_arrays_input_primary["sir"][local_idx]),
                            "input_primary_sar": float(metric_arrays_input_primary["sar"][local_idx]),
                            "residual_primary_sdr": float(residual_metrics["infer_sdr"][0].item()),
                            "residual_primary_sir": float(residual_metrics["infer_sir"][0].item()),
                            "residual_primary_sar": float(residual_metrics["infer_sar"][0].item()),
                            "component_sum_max_abs_error": float(
                                torch.max(
                                    torch.abs(
                                        single_pred[0, 0]
                                        - (
                                            components["signal_component"][0, 0]
                                            + components["interference_component"][0, 0]
                                            + components["artifact_component"][0, 0]
                                        )
                                    )
                                ).item()
                            ),
                        }
                        if args.nmr and pred_nmr_rows is not None:
                            example_row.update(pred_nmr_rows[local_idx])
                        example_rows.append(example_row)

                sample_handle.flush()
                raw_handle.flush()
                if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
                    print(
                        f"[case-studies] {case_def['case_tag']} processed batch {batch_idx}/{total_batches}",
                        flush=True,
                    )

        _write_summary_csv(case_dir / "summary_metrics.csv", summary_acc, case_def, args.num_steps)
        example_export_dir = ""
        if export_examples:
            example_export_path = case_dir / f"eval_examples_{args.split}_adjacent_{args.num_example_exports}samples_{args.num_steps}steps"
            _write_example_export(
                example_dir=example_export_path,
                case_def=case_def,
                cfg=cfg,
                config_path=args.config.resolve(),
                checkpoint_path=args.checkpoint.resolve(),
                checkpoint_epoch=checkpoint_epoch,
                split=args.split,
                num_steps=args.num_steps,
                disable_model_autocast=disable_model_autocast,
                selection_seed=selection_seed,
                nmr_enabled=args.nmr,
                nmr_settings={
                    "peak_prominence_threshold": args.nmr_peak_prominence_threshold,
                    "peak_height_threshold": args.nmr_peak_height_threshold,
                    "peak_match_tolerance_points": args.nmr_peak_match_tolerance_points,
                    "peak_intensity_match_fraction": args.nmr_peak_intensity_match_fraction,
                },
                example_rows=example_rows,
            )
            example_export_dir = str(example_export_path.resolve())

        case_manifest = {
            "config_path": str(args.config.resolve()),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_epoch": checkpoint_epoch,
            "split_used": args.split,
            "num_steps": int(args.num_steps),
            "selection_seed": selection_seed,
            "sampling_seed": sampling_seed,
            "nmr_enabled": bool(args.nmr),
            "case_tag": case_def["case_tag"],
            "case_label": case_def["case_label"],
            "case_description": case_def["description"],
            "num_samples": len(records),
            "component_roles": list(case_def["component_roles"]),
            "component_weights": list(case_def["weights"]),
            "recursive_second_pass": recursive_second_pass,
            "full_outputs_dir": str((case_dir / "full_outputs_chunks").resolve()),
            "sample_manifest_csv": str((case_dir / "sample_manifest.csv").resolve()),
            "raw_metrics_csv": str((case_dir / "raw_metrics.csv").resolve()),
            "summary_metrics_csv": str((case_dir / "summary_metrics.csv").resolve()),
            "example_export_dir": example_export_dir,
        }
        if recursive_second_pass:
            case_manifest["stored_arrays"] = [
                "pure_primary_spectra",
                "pure_intermediate_spectra",
                "pure_minority_spectra",
                "input_spectra",
                "initial_prediction_spectra",
                "initial_residual_spectra",
                "scaled_prediction_input_spectra",
                "scaled_residual_input_spectra",
                "prediction_prediction_spectra",
                "residual_prediction_spectra",
                "prediction_input_scale",
                "residual_input_scale",
            ]
        with open(case_dir / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(case_manifest, handle, indent=2)

        top_manifest["cases"].append(case_manifest)

    _merge_case_manifest(args.output_root, top_manifest)
    print("[case-studies] complete", flush=True)


if __name__ == "__main__":
    run_case_studies(parse_args())
