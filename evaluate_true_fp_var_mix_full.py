"""Evaluate a variable-composition checkpoint at one target fraction."""

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from Train_fp_var_mix import _compute_inference_metrics, instantiate_model, read_smiles_list
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


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    return repo_root / "configs" / "variable_ir.yaml"


def _unwrap_callable(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


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


def _build_adjacent_cycle_pairs(num_items: int, selection_seed: int):
    generator = torch.Generator().manual_seed(int(selection_seed))
    order = torch.randperm(num_items, generator=generator).tolist()
    pairs = []
    for sample_index in range(num_items):
        target_index = order[sample_index]
        primary_index = order[(sample_index + 1) % num_items]
        pairs.append((sample_index, target_index, primary_index))
    return order, pairs


def _prepare_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("manifest.json", "sample_pairs.csv", "raw_metrics.csv", "summary_metrics.csv"):
        path = output_dir / filename
        if path.exists():
            path.unlink()


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


def _set_sampling_seed(seed: int):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _metric_arrays(metrics: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "mse": metrics["infer_mse"].detach().cpu().numpy().astype(np.float64),
        "mae": metrics["infer_mae"].detach().cpu().numpy().astype(np.float64),
        "sdr": metrics["infer_sdr"].detach().cpu().numpy().astype(np.float64),
        "sir": metrics["infer_sir"].detach().cpu().numpy().astype(np.float64),
        "sar": metrics["infer_sar"].detach().cpu().numpy().astype(np.float64),
        "peak_abs_error": metrics["infer_peak_abs_error"].detach().cpu().numpy().astype(np.float64),
    }


def _suppress_bss_metrics_for_alpha_zero(metrics: dict[str, torch.Tensor], minority_alpha: float):
    if float(minority_alpha) != 0.0:
        return metrics
    updated = dict(metrics)
    for key in ("infer_sdr", "infer_sir", "infer_sar"):
        updated[key] = torch.full_like(updated[key], torch.nan)
    return updated


def _write_pairs_csv(path: Path, pair_rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
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


def _write_summary_csv(path: Path, summary_rows):
    fieldnames = [
        "condition",
        "steps",
        "count",
    ]
    for metric_name in SUMMARY_METRIC_NAMES:
        fieldnames.extend(
            [
                f"mean_{metric_name}",
                f"median_{metric_name}",
                f"std_{metric_name}",
                f"finite_{metric_name}_fraction",
            ]
        )
    for row in summary_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _build_summary_rows(summary_acc, minority_alpha: float, nmr_enabled: bool, nmr_fieldnames):
    summary_rows = []
    for (condition, steps), metric_lists in sorted(summary_acc.items(), key=lambda item: (item[0][0], item[0][1])):
        row = {
            "condition": condition,
            "steps": int(steps),
            "count": len(metric_lists["mse"]),
            "minority_alpha": float(minority_alpha),
        }
        for metric_name in SUMMARY_METRIC_NAMES:
            values = np.asarray(metric_lists[metric_name], dtype=np.float64)
            finite = np.isfinite(values)
            row[f"mean_{metric_name}"] = float(np.mean(values))
            row[f"median_{metric_name}"] = float(np.median(values))
            row[f"std_{metric_name}"] = float(np.std(values))
            row[f"finite_{metric_name}_fraction"] = float(np.mean(finite))
        if nmr_enabled:
            for metric_name in nmr_fieldnames:
                values = np.asarray(metric_lists[metric_name], dtype=np.float64)
                row[f"mean_{metric_name}"] = float(np.mean(values)) if values.size > 0 else ""
        summary_rows.append(row)
    return summary_rows


def _build_variable_mixture_batch(
    split_spectra: torch.Tensor,
    target_indices: torch.Tensor,
    primary_indices: torch.Tensor,
    minority_alpha: float,
):
    target_raw = split_spectra.index_select(0, target_indices).unsqueeze(1)
    primary_raw = split_spectra.index_select(0, primary_indices).unsqueeze(1)
    pure_target = _normalize_batch_zero_one(target_raw)
    pure_primary = _normalize_batch_zero_one(primary_raw)

    alpha = torch.full(
        (target_raw.shape[0], 1, 1),
        float(minority_alpha),
        dtype=target_raw.dtype,
    )
    primary_alpha = 1.0 - alpha

    target_weighted = alpha * pure_target
    primary_weighted = primary_alpha * pure_primary
    mixture_pre = target_weighted + primary_weighted
    mixture, mixture_scale = _normalize_batch_by_max(mixture_pre)
    scaled_target = target_weighted / mixture_scale
    scaled_primary = primary_weighted / mixture_scale
    alpha_prime = alpha / mixture_scale
    target_max_intensity = scaled_target.amax(dim=(1, 2))

    return {
        "target_raw": target_raw,
        "primary_raw": primary_raw,
        "pure_target": pure_target,
        "pure_primary": pure_primary,
        "target_weighted": target_weighted,
        "primary_weighted": primary_weighted,
        "mixture_pre": mixture_pre,
        "mixture": mixture,
        "scaled_target": scaled_target,
        "scaled_primary": scaled_primary,
        "mixture_scale": mixture_scale.view(-1),
        "alpha_prime": alpha_prime.view(-1),
        "target_max_intensity": target_max_intensity,
    }


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
            }
        )
    return metric_rows


def evaluate(args):
    cfg = OmegaConf.load(args.config)
    evaluation_cfg = cfg.get("evaluation", {})
    args.split = args.split or evaluation_cfg.get("split", None)
    if args.split not in ("val", "test"):
        raise ValueError("Set evaluation.split to 'val' or 'test' in the config.")
    if args.steps is None:
        args.steps = evaluation_cfg.get("steps", [10, 100])
    if isinstance(args.steps, (int, float)):
        args.steps = [int(args.steps)]
    if args.minority_alpha is None:
        args.minority_alpha = evaluation_cfg.get("minority_alpha", None)
    if args.minority_alpha is None:
        raise ValueError("Set evaluation.minority_alpha in the config.")
    if args.device is None:
        args.device = evaluation_cfg.get("device", None)
    if args.nmr is None:
        args.nmr = bool(evaluation_cfg.get("nmr", False))
    configured_checkpoint = evaluation_cfg.get("checkpoint_path", None)
    configured_output = evaluation_cfg.get("output_dir", None)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else (
            Path(configured_checkpoint)
            if configured_checkpoint
            else Path(cfg.work_dir) / f"{cfg.model_save_prefix}_best.pt"
        )
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            Path(configured_output)
            if configured_output
            else Path(cfg.work_dir)
            / "results"
            / f"alpha_{args.minority_alpha:.2f}".replace(".", "p")
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
    steps_list = [int(step) for step in args.steps]
    max_samples = int(args.max_samples) if args.max_samples is not None else None
    minority_alpha = float(args.minority_alpha)
    nmr_enabled = bool(args.nmr)

    if not (0.0 <= minority_alpha <= 0.5):
        raise ValueError(f"--minority-alpha must be in [0, 0.5], got {minority_alpha}")

    print(f"Loading config from {args.config}")
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Writing evaluation outputs to {output_dir}")
    print(
        f"Using split={args.split} device={device} batch_size={batch_size} "
        f"selection_seed={selection_seed} sampling_seed={sampling_seed} steps={steps_list} "
        f"minority_alpha={minority_alpha:.4f}"
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
    order, pairs = _build_adjacent_cycle_pairs(split_spectra.shape[0], selection_seed)
    if max_samples is not None:
        pairs = pairs[:max_samples]

    _prepare_output_dir(output_dir)

    pair_rows = []
    for sample_index, target_index, primary_index in pairs:
        pair_rows.append(
            {
                "sample_index": sample_index,
                "target_index": target_index,
                "primary_index": primary_index,
                "target_smiles": split_smiles[target_index],
                "primary_smiles": split_smiles[primary_index],
            }
        )
    _write_pairs_csv(output_dir / "sample_pairs.csv", pair_rows)

    raw_csv_path = output_dir / "raw_metrics.csv"
    raw_fieldnames = [
        "sample_index",
        "condition",
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
    ]
    if nmr_enabled:
        raw_fieldnames.extend(nmr_fieldnames)
    summary_acc = {}

    with open(raw_csv_path, "w", encoding="utf-8", newline="") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fieldnames)
        raw_writer.writeheader()

        def write_metric_block(
            sample_indices,
            condition,
            steps,
            metrics,
            target_max_intensity,
            mixture_scale,
            nmr_rows=None,
        ):
            arrays = _metric_arrays(metrics)
            key = (condition, int(steps))
            block = summary_acc.setdefault(
                key,
                {
                    metric_name: []
                    for metric_name in SUMMARY_METRIC_NAMES
                }
                | (
                    {name: [] for name in nmr_fieldnames}
                    if nmr_enabled
                    else {}
                ),
            )
            target_max_np = target_max_intensity.detach().cpu().numpy().astype(np.float64)
            mixture_scale_np = mixture_scale.detach().cpu().numpy().astype(np.float64)
            for local_idx, sample_index in enumerate(sample_indices):
                row = {
                    "sample_index": int(sample_index),
                    "condition": condition,
                    "steps": int(steps),
                    "minority_alpha": float(minority_alpha),
                    "target_max_intensity": float(target_max_np[local_idx]),
                    "mixture_scale": float(mixture_scale_np[local_idx]),
                }
                for metric_name in SUMMARY_METRIC_NAMES:
                    row[metric_name] = float(arrays[metric_name][local_idx])
                    block[metric_name].append(float(arrays[metric_name][local_idx]))
                if nmr_enabled:
                    if nmr_rows is not None:
                        row.update(nmr_rows[local_idx])
                        for name in nmr_fieldnames:
                            block[name].append(float(nmr_rows[local_idx][name]))
                    else:
                        for name in nmr_fieldnames:
                            row[name] = ""
                raw_writer.writerow(row)

        def flush_partial_outputs():
            raw_handle.flush()
            summary_rows = _build_summary_rows(
                summary_acc=summary_acc,
                minority_alpha=minority_alpha,
                nmr_enabled=nmr_enabled,
                nmr_fieldnames=nmr_fieldnames,
            )
            _write_summary_csv(output_dir / "summary_metrics.csv", summary_rows)

        total_batches = (len(pairs) + batch_size - 1) // batch_size

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
                minority_alpha=minority_alpha,
            )

            input_target_metrics = _compute_inference_metrics(
                batch["mixture"],
                batch["scaled_target"],
                batch["scaled_primary"],
                pure_target=batch["pure_target"],
            )
            input_target_metrics = _suppress_bss_metrics_for_alpha_zero(input_target_metrics, minority_alpha)
            input_primary_metrics = _compute_inference_metrics(
                batch["mixture"],
                batch["scaled_primary"],
                batch["scaled_target"],
                pure_target=batch["pure_primary"],
            )
            input_primary_metrics = _suppress_bss_metrics_for_alpha_zero(input_primary_metrics, minority_alpha)
            write_metric_block(
                sample_indices,
                "input_target_reference",
                0,
                input_target_metrics,
                batch["target_max_intensity"],
                batch["mixture_scale"],
            )
            write_metric_block(
                sample_indices,
                "input_primary_reference",
                0,
                input_primary_metrics,
                batch["target_max_intensity"],
                batch["mixture_scale"],
            )
            flush_partial_outputs()

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
                print(f"[reference] processed batch {batch_idx}/{total_batches}")

        for num_steps in steps_list:
            _set_sampling_seed(sampling_seed)
            print(f"[prediction_target] starting num_steps={num_steps}")
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
                    minority_alpha=minority_alpha,
                )

                mix_device = batch["mixture"].to(device)
                target_device = batch["scaled_target"].to(device)
                primary_device = batch["scaled_primary"].to(device)
                pure_target_device = batch["pure_target"].to(device)

                if disable_model_autocast:
                    pred, _ = _sample_without_model_autocast(model, mix_device, num_steps=num_steps)
                else:
                    with torch.no_grad():
                        pred, _ = model.sample(mix_device, num_steps=num_steps, show_progress=False)

                pred = pred.unsqueeze(1).detach().cpu()
                target_cpu = target_device.detach().cpu()
                primary_cpu = primary_device.detach().cpu()

                pred_target_metrics = _compute_inference_metrics(
                    pred,
                    target_cpu,
                    primary_cpu,
                    pure_target=pure_target_device.detach().cpu(),
                )
                pred_target_metrics = _suppress_bss_metrics_for_alpha_zero(pred_target_metrics, minority_alpha)
                pred_target_nmr_rows = None
                if nmr_enabled:
                    pred_target_nmr_rows = _compute_nmr_peak_metric_rows(
                        pred_batch=pred,
                        target_batch=target_cpu,
                        prominence_threshold=args.nmr_peak_prominence_threshold,
                        height_threshold=args.nmr_peak_height_threshold,
                        tolerance_points=args.nmr_peak_match_tolerance_points,
                        adjacent_peak_valley_fraction=args.nmr_adjacent_peak_valley_fraction,
                        peak_intensity_match_fraction=args.nmr_peak_intensity_match_fraction,
                    )
                write_metric_block(
                    sample_indices,
                    "prediction_target",
                    num_steps,
                    pred_target_metrics,
                    batch["target_max_intensity"],
                    batch["mixture_scale"],
                    nmr_rows=pred_target_nmr_rows,
                )

                residual_primary = _normalize_batch_zero_one(batch["mixture"] - pred)
                residual_primary_metrics = _compute_inference_metrics(
                    residual_primary,
                    batch["pure_primary"],
                    batch["pure_target"],
                    pure_target=batch["pure_primary"],
                )
                residual_primary_metrics = _suppress_bss_metrics_for_alpha_zero(residual_primary_metrics, minority_alpha)
                write_metric_block(
                    sample_indices,
                    "residual_primary",
                    num_steps,
                    residual_primary_metrics,
                    batch["target_max_intensity"],
                    batch["mixture_scale"],
                )
                flush_partial_outputs()

                if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
                    print(f"[num_steps={num_steps}] processed batch {batch_idx}/{total_batches}")

    summary_rows = _build_summary_rows(
        summary_acc=summary_acc,
        minority_alpha=minority_alpha,
        nmr_enabled=nmr_enabled,
        nmr_fieldnames=nmr_fieldnames,
    )
    _write_summary_csv(output_dir / "summary_metrics.csv", summary_rows)

    manifest = {
        "config_path": str(args.config),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model_save_prefix": str(cfg.model_save_prefix),
        "split_used": str(args.split),
        "selection_seed": int(selection_seed),
        "sampling_seed": int(sampling_seed),
        "steps": steps_list,
        "num_samples": int(len(pairs)),
        "batch_size": int(batch_size),
        "disable_model_autocast": bool(disable_model_autocast),
        "nmr_enabled": bool(nmr_enabled),
        "minority_alpha": float(minority_alpha),
        "device_used": str(device),
        "pairing_notes": (
            "Pairs are formed by shuffling the split once, then pairing each item "
            "with the next item in the shuffled order, wrapping at the end."
        ),
        "mixture_notes": {
            "construction": (
                "Each mixture is built as minority_alpha * pure_target + "
                "(1 - minority_alpha) * pure_primary, then divided by its maximum value."
            ),
            "prediction_target": "The model prediction is evaluated against the scaled minority target.",
            "input_target_reference": "The normalized mixture is evaluated directly against the scaled minority target.",
            "input_primary_reference": "The normalized mixture is evaluated directly against the scaled primary target.",
            "residual_primary": (
                "Residual primary estimate formed as normalize_0_1(mixture_spectrum - prediction_spectrum), "
                "then evaluated against the pure primary spectrum."
            ),
            "target_max_intensity": "Maximum intensity of the scaled minority target within the normalized mixture frame.",
            "peak_abs_error": "Absolute difference between the predicted and true target maximum intensities.",
        },
    }
    if nmr_enabled:
        manifest["nmr_metric_notes"] = {
            "implementation_source": "nmr_peak_metrics.match_nmr_peaks",
            "intensity_frame": (
                "Prediction and target are independently clipped at zero and "
                "scaled to unit maximum before peak detection."
            ),
            "prediction_target_only": True,
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
        }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote evaluation results to {output_dir}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")


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
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <work_dir>/results/<alpha_tag>/full_eval_val.",
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
        nargs="+",
        default=None,
        help="Inference step counts. Defaults to evaluation.steps in the config.",
    )
    parser.add_argument(
        "--minority-alpha",
        type=float,
        default=None,
        help="Minority weight. Defaults to evaluation.minority_alpha in the config.",
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
    evaluate(parse_args())
