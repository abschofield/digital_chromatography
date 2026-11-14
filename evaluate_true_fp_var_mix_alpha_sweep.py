"""Evaluate a variable-composition checkpoint over a range of target fractions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from Train_fp_var_mix import _compute_inference_metrics, instantiate_model
from evaluate_true_fp_var_mix_full import (
    _build_adjacent_cycle_pairs,
    _build_variable_mixture_batch,
    _default_config_path,
    _load_dataset_split,
    _sample_without_model_autocast,
    _set_sampling_seed,
    _suppress_bss_metrics_for_alpha_zero,
)
from nmr_peak_metrics import (
    DEFAULT_PEAK_HEIGHT_THRESHOLD,
    DEFAULT_PEAK_INTENSITY_MATCH_FRACTION,
    DEFAULT_PEAK_MATCH_TOLERANCE_POINTS,
    DEFAULT_PEAK_PROMINENCE_THRESHOLD,
    match_nmr_peaks,
    normalize_nmr_spectrum,
)


IR_SUMMARY_METRICS = (
    "mae",
    "sdr",
    "sir",
    "sar",
)

NMR_SUMMARY_METRICS = (
    "mae",
    "sdr",
    "precision",
    "recall",
    "tp_peak_intensity_mae",
    "all_predicted_peak_intensity_mae",
)

NMR_MEAN_ONLY_METRICS = (
    "false_positive_rate",
)


def _alpha_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("--alpha-step must be positive.")
    count = int(round((stop - start) / step)) + 1
    values = [round(start + idx * step, 2) for idx in range(count)]
    filtered = [value for value in values if value <= stop + 1e-8]
    if not filtered:
        raise ValueError("No alpha values were generated.")
    return filtered


def _alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace("-", "m").replace(".", "p")


def _prepare_root_output_dir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "sample_pairs.csv",
        "alpha_sweep_summary.csv",
        "alpha_sweep_manifest.json",
    ):
        path = root / filename
        if path.exists():
            path.unlink()


def _prepare_alpha_output_dir(alpha_dir: Path):
    alpha_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("raw_metrics.csv", "summary_metrics.csv", "manifest.json"):
        path = alpha_dir / filename
        if path.exists():
            path.unlink()


def _write_pairs_csv(path: Path, pair_rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "target_index",
                "primary_index",
                "target_smiles",
                "primary_smiles",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)


def _finite_mean(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return ""
    return float(np.mean(arr[finite]))


def _finite_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return ""
    return float(np.std(arr[finite]))


def _compute_nmr_peak_metric_rows(
    pred_batch: torch.Tensor,
    target_batch: torch.Tensor,
    prominence_threshold: float,
    height_threshold: float,
    tolerance_points: int,
    adjacent_peak_valley_fraction: float | None,
    peak_intensity_match_fraction: float | None,
):
    pred_np = pred_batch.detach().cpu().numpy()
    target_np = target_batch.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[:, 0, :]
    if target_np.ndim == 3:
        target_np = target_np[:, 0, :]

    metric_rows = []
    for pred_signal, target_signal in zip(pred_np, target_np):
        pred_signal = normalize_nmr_spectrum(pred_signal)
        target_signal = normalize_nmr_spectrum(target_signal)
        peak_match = match_nmr_peaks(
            target_signal=target_signal,
            prediction_signal=pred_signal,
            prominence_threshold=prominence_threshold,
            height_threshold=height_threshold,
            tolerance_points=tolerance_points,
            adjacent_peak_valley_fraction=adjacent_peak_valley_fraction,
            peak_intensity_match_fraction=peak_intensity_match_fraction,
        )
        pred_clipped = pred_signal
        target_clipped = target_signal

        tp_errors = [
            abs(float(pred_clipped[pred_peak]) - float(target_clipped[target_peak]))
            for target_peak, pred_peak in peak_match["matched_pairs"]
        ]

        all_predicted_errors = []
        for pred_peak, interval in zip(
            peak_match["prediction_peaks"],
            peak_match["prediction_group_intervals"],
        ):
            start, end = int(interval[0]), int(interval[1])
            target_region_max = float(np.max(target_clipped[start : end + 1]))
            pred_height = float(pred_clipped[int(pred_peak)])
            all_predicted_errors.append(abs(pred_height - target_region_max))

        metric_rows.append(
            {
                "precision": float(peak_match["precision"]),
                "recall": float(peak_match["recall"]),
                "total_target_peaks": int(peak_match["total_target_peaks"]),
                "predicted_peaks": int(len(peak_match["prediction_peaks"])),
                "true_positive_count": int(peak_match["true_positive_count"]),
                "underpowered_count": int(peak_match["underpowered_count"]),
                "false_negative_count": int(peak_match["false_negative_count"]),
                "false_positive_count": int(peak_match["false_positive_count"]),
                "false_positive_rate": float(peak_match["false_positive_count"]),
                "tp_peak_intensity_mae": (
                    float(np.mean(tp_errors)) if tp_errors else float("nan")
                ),
                "all_predicted_peak_intensity_mae": (
                    float(np.mean(all_predicted_errors))
                    if all_predicted_errors
                    else float("nan")
                ),
            }
        )
    return metric_rows


def _write_summary_csv(path: Path, summary_row: dict):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)


def _write_alpha_manifest(
    path: Path,
    *,
    args,
    cfg,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    alpha: float,
    num_samples: int,
    nmr_enabled: bool,
    spectrum_length: int,
):
    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "model_save_prefix": str(cfg.model_save_prefix),
        "split_used": str(args.split),
        "selection_seed": int(args.selection_seed if args.selection_seed is not None else cfg.seed),
        "sampling_seed": int(args.sampling_seed if args.sampling_seed is not None else cfg.seed),
        "steps": [int(args.steps)],
        "num_samples": int(num_samples),
        "batch_size": int(args.batch_size if args.batch_size is not None else cfg.batch_size),
        "disable_model_autocast": bool(
            args.disable_model_autocast
            if args.disable_model_autocast is not None
            else cfg.train.get("disable_model_autocast", False)
        ),
        "minority_alpha": float(alpha),
        "nmr_enabled": bool(nmr_enabled),
        "metric_scope": "prediction_target_only",
    }
    if args.save_predictions:
        payload["prediction_archive"] = {
            "file": "prediction_spectra.npy",
            "dtype": "float32",
            "shape": [int(num_samples), int(spectrum_length)],
            "row_order": "sample_index order in the root sample_pairs.csv",
            "intensity_frame": "normalized mixture frame, identical to prediction_target metrics",
        }
    if nmr_enabled:
        payload["nmr_metric_notes"] = {
            "implementation_source": "nmr_peak_metrics.match_nmr_peaks",
            "intensity_frame": (
                "Prediction and target are independently clipped at zero and "
                "scaled to unit maximum before peak detection."
            ),
            "prominence_threshold": float(args.nmr_peak_prominence_threshold),
            "height_threshold": float(args.nmr_peak_height_threshold),
            "tolerance_points": int(args.nmr_peak_match_tolerance_points),
            "adjacent_peak_valley_fraction": float(args.nmr_adjacent_peak_valley_fraction),
            "peak_intensity_match_fraction": float(args.nmr_peak_intensity_match_fraction),
            "precision_definition": (
                "supported_prediction_count / predicted_peaks, where support is defined by "
                "interval overlap with target peak regions."
            ),
            "recall_definition": (
                "true_positive_count / total_target_peaks using the overlap-based NMR criterion."
            ),
            "false_positive_rate_definition": (
                "Average number of false-positive peaks per sample; at the per-sample "
                "level this is identical to false_positive_count."
            ),
            "tp_peak_intensity_mae_definition": (
                "Mean absolute difference between matched true-positive peak heights."
            ),
            "all_predicted_peak_intensity_mae_definition": (
                "Mean absolute difference between each predicted peak height and the "
                "maximum target intensity within that predicted peak region."
            ),
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_alpha_sweep(args):
    cfg = OmegaConf.load(args.config)
    evaluation_cfg = cfg.get("evaluation", {})
    args.split = args.split or evaluation_cfg.get("split", None)
    if args.split not in ("val", "test"):
        raise ValueError("Set evaluation.split to 'val' or 'test' in the config.")
    args.steps = int(
        args.steps
        if args.steps is not None
        else evaluation_cfg.get("alpha_sweep_steps", 10)
    )
    args.alpha_start = float(
        args.alpha_start
        if args.alpha_start is not None
        else evaluation_cfg.get("alpha_start", 0.0)
    )
    args.alpha_stop = float(
        args.alpha_stop
        if args.alpha_stop is not None
        else evaluation_cfg.get("alpha_stop", 0.4)
    )
    args.alpha_step = float(
        args.alpha_step
        if args.alpha_step is not None
        else evaluation_cfg.get("alpha_step", 0.01)
    )
    if args.device is None:
        args.device = evaluation_cfg.get("device", None)
    if args.nmr is None:
        args.nmr = bool(evaluation_cfg.get("nmr", False))
    if args.save_predictions is None:
        args.save_predictions = bool(evaluation_cfg.get("save_predictions", False))
    configured_checkpoint = evaluation_cfg.get("checkpoint_path", None)
    configured_output = evaluation_cfg.get("alpha_sweep_output_dir", None)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else (
            Path(configured_checkpoint)
            if configured_checkpoint
            else Path(cfg.work_dir) / f"{cfg.model_save_prefix}_best.pt"
        )
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else (
            Path(configured_output)
            if configured_output
            else Path(cfg.work_dir) / "alpha_sweep_eval_10steps"
        )
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    disable_model_autocast = bool(
        args.disable_model_autocast
        if args.disable_model_autocast is not None
        else cfg.train.get("disable_model_autocast", False)
    )
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.batch_size)
    selection_seed = int(args.selection_seed if args.selection_seed is not None else cfg.seed)
    sampling_seed = int(args.sampling_seed if args.sampling_seed is not None else cfg.seed)
    alpha_values = _alpha_values(args.alpha_start, args.alpha_stop, args.alpha_step)
    nmr_enabled = bool(args.nmr)

    if args.save_predictions and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Prediction archive requires an empty output directory: {output_root}"
        )

    print(f"Loading config from {args.config}")
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Writing alpha-sweep outputs to {output_root}")
    print(
        f"Using split={args.split} device={device} batch_size={batch_size} "
        f"selection_seed={selection_seed} sampling_seed={sampling_seed} "
        f"steps={[int(args.steps)]} alpha_values={alpha_values[:3]} ... {alpha_values[-3:]}"
    )
    if nmr_enabled:
        print(
            "Using NMR peak metrics:"
            f" prominence>={args.nmr_peak_prominence_threshold:.4f}"
            f" height>={args.nmr_peak_height_threshold:.4f}"
            f" tolerance_points={args.nmr_peak_match_tolerance_points}"
            f" adjacent_peak_valley_fraction={args.nmr_adjacent_peak_valley_fraction:.4f}"
            f" peak_intensity_match_fraction={args.nmr_peak_intensity_match_fraction:.4f}"
        )

    model = instantiate_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device)
    model.eval()

    split_spectra, split_smiles = _load_dataset_split(cfg, args.split)
    _, pairs = _build_adjacent_cycle_pairs(split_spectra.shape[0], selection_seed)
    if args.max_samples is not None:
        pairs = pairs[: int(args.max_samples)]

    _prepare_root_output_dir(output_root)
    pair_rows = [
        {
            "sample_index": sample_index,
            "target_index": target_index,
            "primary_index": primary_index,
            "target_smiles": split_smiles[target_index],
            "primary_smiles": split_smiles[primary_index],
        }
        for sample_index, target_index, primary_index in pairs
    ]
    _write_pairs_csv(output_root / "sample_pairs.csv", pair_rows)

    alpha_summary_rows = []
    raw_fieldnames = [
        "sample_index",
        "steps",
        "minority_alpha",
        "target_max_intensity",
        "mixture_scale",
        "mse",
        "mae",
        "sdr",
        "sir",
        "sar",
        "peak_abs_error",
    ]
    nmr_fieldnames = [
        "precision",
        "recall",
        "total_target_peaks",
        "predicted_peaks",
        "true_positive_count",
        "underpowered_count",
        "false_negative_count",
        "false_positive_count",
        "false_positive_rate",
        "tp_peak_intensity_mae",
        "all_predicted_peak_intensity_mae",
    ]
    if nmr_enabled:
        raw_fieldnames.extend(nmr_fieldnames)

    total_batches = (len(pairs) + batch_size - 1) // batch_size

    for alpha in alpha_values:
        alpha_dir = output_root / _alpha_tag(alpha)
        _prepare_alpha_output_dir(alpha_dir)
        prediction_archive = None
        if args.save_predictions:
            prediction_archive = np.lib.format.open_memmap(
                alpha_dir / "prediction_spectra.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(pairs), int(split_spectra.shape[1])),
            )
        print(f"[alpha={alpha:.2f}] starting evaluation")
        _set_sampling_seed(sampling_seed)

        raw_rows = []
        metric_acc = {
            metric_name: []
            for metric_name in (
                (NMR_SUMMARY_METRICS + NMR_MEAN_ONLY_METRICS)
                if nmr_enabled
                else IR_SUMMARY_METRICS
            )
        }

        for batch_idx, start in enumerate(range(0, len(pairs), batch_size), start=1):
            end = min(len(pairs), start + batch_size)
            batch_pairs = pairs[start:end]
            sample_indices = [row[0] for row in batch_pairs]
            target_indices = torch.tensor([row[1] for row in batch_pairs], dtype=torch.long)
            primary_indices = torch.tensor([row[2] for row in batch_pairs], dtype=torch.long)

            batch = _build_variable_mixture_batch(
                split_spectra=split_spectra,
                target_indices=target_indices,
                primary_indices=primary_indices,
                minority_alpha=float(alpha),
            )

            mix_device = batch["mixture"].to(device)
            target_device = batch["scaled_target"].to(device)
            primary_device = batch["scaled_primary"].to(device)
            pure_target_device = batch["pure_target"].to(device)

            if disable_model_autocast:
                pred, _ = _sample_without_model_autocast(model, mix_device, num_steps=int(args.steps))
            else:
                with torch.no_grad():
                    pred, _ = model.sample(mix_device, num_steps=int(args.steps), show_progress=False)

            pred = pred.unsqueeze(1).detach().cpu()
            target_cpu = target_device.detach().cpu()
            primary_cpu = primary_device.detach().cpu()
            if prediction_archive is not None:
                prediction_archive[start:end] = pred[:, 0, :].numpy()

            pred_target_metrics = _compute_inference_metrics(
                pred,
                target_cpu,
                primary_cpu,
                pure_target=pure_target_device.detach().cpu(),
            )
            pred_target_metrics = _suppress_bss_metrics_for_alpha_zero(
                pred_target_metrics,
                float(alpha),
            )

            arrays = {
                "mse": pred_target_metrics["infer_mse"].detach().cpu().numpy().astype(np.float64),
                "mae": pred_target_metrics["infer_mae"].detach().cpu().numpy().astype(np.float64),
                "sdr": pred_target_metrics["infer_sdr"].detach().cpu().numpy().astype(np.float64),
                "sir": pred_target_metrics["infer_sir"].detach().cpu().numpy().astype(np.float64),
                "sar": pred_target_metrics["infer_sar"].detach().cpu().numpy().astype(np.float64),
                "peak_abs_error": pred_target_metrics["infer_peak_abs_error"].detach().cpu().numpy().astype(np.float64),
            }
            nmr_rows = None
            if nmr_enabled:
                nmr_rows = _compute_nmr_peak_metric_rows(
                    pred_batch=pred,
                    target_batch=target_cpu,
                    prominence_threshold=args.nmr_peak_prominence_threshold,
                    height_threshold=args.nmr_peak_height_threshold,
                    tolerance_points=args.nmr_peak_match_tolerance_points,
                    adjacent_peak_valley_fraction=args.nmr_adjacent_peak_valley_fraction,
                    peak_intensity_match_fraction=args.nmr_peak_intensity_match_fraction,
                )

            target_max_np = batch["target_max_intensity"].detach().cpu().numpy().astype(np.float64)
            mixture_scale_np = batch["mixture_scale"].detach().cpu().numpy().astype(np.float64)
            for local_idx, sample_index in enumerate(sample_indices):
                row = {
                    "sample_index": int(sample_index),
                    "steps": int(args.steps),
                    "minority_alpha": float(alpha),
                    "target_max_intensity": float(target_max_np[local_idx]),
                    "mixture_scale": float(mixture_scale_np[local_idx]),
                    "mse": float(arrays["mse"][local_idx]),
                    "mae": float(arrays["mae"][local_idx]),
                    "sdr": float(arrays["sdr"][local_idx]),
                    "sir": float(arrays["sir"][local_idx]),
                    "sar": float(arrays["sar"][local_idx]),
                    "peak_abs_error": float(arrays["peak_abs_error"][local_idx]),
                }
                metric_acc["mae"].append(row["mae"])
                metric_acc["sdr"].append(row["sdr"])
                if not nmr_enabled:
                    metric_acc["sir"].append(row["sir"])
                    metric_acc["sar"].append(row["sar"])
                if nmr_enabled and nmr_rows is not None:
                    row.update(nmr_rows[local_idx])
                    metric_acc["precision"].append(float(nmr_rows[local_idx]["precision"]))
                    metric_acc["recall"].append(float(nmr_rows[local_idx]["recall"]))
                    metric_acc["false_positive_rate"].append(
                        float(nmr_rows[local_idx]["false_positive_rate"])
                    )
                    metric_acc["tp_peak_intensity_mae"].append(
                        float(nmr_rows[local_idx]["tp_peak_intensity_mae"])
                    )
                    metric_acc["all_predicted_peak_intensity_mae"].append(
                        float(nmr_rows[local_idx]["all_predicted_peak_intensity_mae"])
                    )
                raw_rows.append(row)

            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
                print(f"[alpha={alpha:.2f}] processed batch {batch_idx}/{total_batches}")

        if prediction_archive is not None:
            prediction_archive.flush()
            del prediction_archive

        with (alpha_dir / "raw_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=raw_fieldnames)
            writer.writeheader()
            writer.writerows(raw_rows)

        summary_row = {
            "alpha": float(alpha),
            "steps": int(args.steps),
            "count": int(len(raw_rows)),
        }
        if nmr_enabled:
            for metric_name in NMR_SUMMARY_METRICS:
                summary_row[f"mean_{metric_name}"] = _finite_mean(metric_acc[metric_name])
                summary_row[f"std_{metric_name}"] = _finite_std(metric_acc[metric_name])
            for metric_name in NMR_MEAN_ONLY_METRICS:
                summary_row[f"mean_{metric_name}"] = _finite_mean(metric_acc[metric_name])
        else:
            for metric_name in IR_SUMMARY_METRICS:
                summary_row[f"mean_{metric_name}"] = _finite_mean(metric_acc[metric_name])
                summary_row[f"std_{metric_name}"] = _finite_std(metric_acc[metric_name])

        _write_summary_csv(alpha_dir / "summary_metrics.csv", summary_row)
        _write_alpha_manifest(
            alpha_dir / "manifest.json",
            args=args,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=int(checkpoint.get("epoch", -1)),
            alpha=float(alpha),
            num_samples=len(raw_rows),
            nmr_enabled=nmr_enabled,
            spectrum_length=int(split_spectra.shape[1]),
        )
        alpha_summary_rows.append(summary_row)
        print(f"[alpha={alpha:.2f}] wrote outputs to {alpha_dir}")

    summary_path = output_root / "alpha_sweep_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alpha_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(alpha_summary_rows)

    sweep_manifest = {
        "config_path": str(args.config),
        "checkpoint_path": str(checkpoint_path),
        "model_save_prefix": str(cfg.model_save_prefix),
        "split_used": str(args.split),
        "steps": [int(args.steps)],
        "alpha_start": float(args.alpha_start),
        "alpha_stop": float(args.alpha_stop),
        "alpha_step": float(args.alpha_step),
        "alpha_values": alpha_values,
        "nmr_enabled": bool(nmr_enabled),
        "num_samples": int(len(pairs)),
        "batch_size": int(batch_size),
        "save_predictions": bool(args.save_predictions),
        "per_alpha_dirs": [str(output_root / _alpha_tag(alpha)) for alpha in alpha_values],
    }
    (output_root / "alpha_sweep_manifest.json").write_text(
        json.dumps(sweep_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote alpha-sweep summary to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. Defaults to <work_dir>/<model_save_prefix>_best.pt.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory for the alpha-sweep outputs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "test"),
        default=None,
        help="Dataset split. Defaults to evaluation.split in the config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps. Defaults to evaluation.alpha_sweep_steps.",
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=None,
        help="Starting alpha. Defaults to evaluation.alpha_start.",
    )
    parser.add_argument(
        "--alpha-stop",
        type=float,
        default=None,
        help="Ending alpha. Defaults to evaluation.alpha_stop.",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=None,
        help="Alpha increment. Defaults to evaluation.alpha_step.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=None,
        help="Seed used to shuffle the split before building adjacent-cycle pairs.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Seed used to make the diffusion sampling comparisons reproducible.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size. Defaults to the config batch_size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests. Defaults to all samples in the split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda or cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--disable-model-autocast",
        dest="disable_model_autocast",
        action="store_true",
        default=None,
        help="Disable the model's decorated autocast wrappers during sampling.",
    )
    parser.add_argument(
        "--keep-model-autocast",
        dest="disable_model_autocast",
        action="store_false",
        help="Keep the model's normal autocast wrappers during sampling.",
    )
    parser.add_argument(
        "--nmr",
        dest="nmr",
        action="store_true",
        default=None,
        help="Enable NMR peak metrics, overriding the config.",
    )
    parser.add_argument(
        "--no-nmr",
        dest="nmr",
        action="store_false",
        help="Disable NMR peak metrics, overriding the config.",
    )
    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        action="store_true",
        default=None,
        help="Save float32 prediction spectra by alpha; requires a new, empty output root.",
    )
    parser.add_argument(
        "--no-save-predictions",
        dest="save_predictions",
        action="store_false",
        help="Do not save prediction arrays, overriding the config.",
    )
    parser.add_argument(
        "--nmr-peak-prominence-threshold",
        type=float,
        default=DEFAULT_PEAK_PROMINENCE_THRESHOLD,
        help="Absolute prominence threshold used for NMR peak detection.",
    )
    parser.add_argument(
        "--nmr-peak-height-threshold",
        type=float,
        default=DEFAULT_PEAK_HEIGHT_THRESHOLD,
        help="Absolute height threshold used for NMR peak detection.",
    )
    parser.add_argument(
        "--nmr-peak-match-tolerance-points",
        type=int,
        default=DEFAULT_PEAK_MATCH_TOLERANCE_POINTS,
        help="Maximum index separation used when building overlap regions for NMR peak matching.",
    )
    parser.add_argument(
        "--nmr-adjacent-peak-valley-fraction",
        type=float,
        default=0.1,
        help="Merge adjacent NMR peaks unless the valley drops below this fraction of the smaller peak.",
    )
    parser.add_argument(
        "--nmr-peak-intensity-match-fraction",
        type=float,
        default=DEFAULT_PEAK_INTENSITY_MATCH_FRACTION if DEFAULT_PEAK_INTENSITY_MATCH_FRACTION is not None else 0.5,
        help="Minimum prediction-region height as a fraction of target peak height for NMR TP credit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_alpha_sweep(parse_args())
