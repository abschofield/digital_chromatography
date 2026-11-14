"""Evaluate a fixed 3:1 checkpoint on a complete validation or test split."""

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from Train_fp32 import _compute_bss_metrics, _normalize_zero_one, instantiate_model, read_smiles_list
from nmr_peak_metrics import (
    DEFAULT_PEAK_HEIGHT_THRESHOLD,
    DEFAULT_PEAK_INTENSITY_MATCH_FRACTION,
    DEFAULT_PEAK_MATCH_TOLERANCE_POINTS,
    DEFAULT_PEAK_PROMINENCE_THRESHOLD,
    match_nmr_peaks,
    normalize_nmr_spectrum,
)


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    return repo_root / "configs" / "fixed_ir.yaml"


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


def _compute_metric_triplet(pred, signal, other):
    sdr, sir, sar = _compute_bss_metrics(pred, signal, other)
    reduction_dims = tuple(range(1, pred.ndim))
    mse = torch.mean((pred - signal) ** 2, dim=reduction_dims)
    mae = torch.mean(torch.abs(pred - signal), dim=reduction_dims)
    return {
        "mse": mse.detach().cpu().numpy().astype(np.float64),
        "mae": mae.detach().cpu().numpy().astype(np.float64),
        "sdr": sdr.detach().cpu().numpy().astype(np.float64),
        "sir": sir.detach().cpu().numpy().astype(np.float64),
        "sar": sar.detach().cpu().numpy().astype(np.float64),
    }


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
        "mean_mse",
        "mean_mae",
        "mean_sdr",
        "mean_sir",
        "mean_sar",
        "median_sdr",
        "median_sir",
        "median_sar",
        "std_sdr",
        "std_sir",
        "std_sar",
        "finite_sdr_fraction",
        "finite_sir_fraction",
        "finite_sar_fraction",
    ]
    for row in summary_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


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
        args.steps = evaluation_cfg.get("steps", [10, 100, 1000])
    if isinstance(args.steps, (int, float)):
        args.steps = [int(args.steps)]
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
            else Path(cfg.work_dir) / f"{cfg.model_save_prefix}_full_eval_{args.split}"
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
    nmr_enabled = bool(args.nmr)

    print(f"Loading config from {args.config}")
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Writing evaluation outputs to {output_dir}")
    print(
        f"Using split={args.split} device={device} batch_size={batch_size} "
        f"selection_seed={selection_seed} sampling_seed={sampling_seed} steps={steps_list}"
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
        "mse",
        "mae",
        "sdr",
        "sir",
        "sar",
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

        def write_metric_block(sample_indices, condition, steps, metrics, nmr_rows=None):
            key = (condition, int(steps))
            block = summary_acc.setdefault(
                key,
                {
                    "mse": [],
                    "mae": [],
                    "sdr": [],
                    "sir": [],
                    "sar": [],
                    **({name: [] for name in nmr_fieldnames} if nmr_enabled else {}),
                },
            )
            for local_idx, sample_index in enumerate(sample_indices):
                mse_val = float(metrics["mse"][local_idx])
                mae_val = float(metrics["mae"][local_idx])
                sdr_val = float(metrics["sdr"][local_idx])
                sir_val = float(metrics["sir"][local_idx])
                sar_val = float(metrics["sar"][local_idx])
                row = {
                    "sample_index": int(sample_index),
                    "condition": condition,
                    "steps": int(steps),
                    "mse": mse_val,
                    "mae": mae_val,
                    "sdr": sdr_val,
                    "sir": sir_val,
                    "sar": sar_val,
                }
                if nmr_enabled:
                    if nmr_rows is not None:
                        row.update(nmr_rows[local_idx])
                        for name in nmr_fieldnames:
                            block[name].append(float(nmr_rows[local_idx][name]))
                    else:
                        for name in nmr_fieldnames:
                            row[name] = ""
                raw_writer.writerow(row)
                block["mse"].append(mse_val)
                block["mae"].append(mae_val)
                block["sdr"].append(sdr_val)
                block["sir"].append(sir_val)
                block["sar"].append(sar_val)

        total_batches = (len(pairs) + batch_size - 1) // batch_size

        for batch_idx, start in enumerate(range(0, len(pairs), batch_size), start=1):
            end = min(len(pairs), start + batch_size)
            batch_pairs = pairs[start:end]
            sample_indices = [row[0] for row in batch_pairs]
            target_indices = torch.tensor([row[1] for row in batch_pairs], dtype=torch.long)
            primary_indices = torch.tensor([row[2] for row in batch_pairs], dtype=torch.long)

            target_raw = split_spectra.index_select(0, target_indices).unsqueeze(1)
            primary_raw = split_spectra.index_select(0, primary_indices).unsqueeze(1)
            target = _normalize_batch_zero_one(target_raw)
            primary = _normalize_batch_zero_one(primary_raw)
            mixture_pre = target + 3.0 * primary
            mixture = _normalize_batch_zero_one(mixture_pre)

            input_target_metrics = _compute_metric_triplet(mixture, target, primary)
            input_primary_metrics = _compute_metric_triplet(mixture, primary, target)
            write_metric_block(sample_indices, "input_target_reference", 0, input_target_metrics)
            write_metric_block(sample_indices, "input_primary_reference", 0, input_primary_metrics)

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

                target_raw = split_spectra.index_select(0, target_indices).unsqueeze(1)
                primary_raw = split_spectra.index_select(0, primary_indices).unsqueeze(1)
                target = _normalize_batch_zero_one(target_raw)
                primary = _normalize_batch_zero_one(primary_raw)
                mixture_pre = target + 3.0 * primary
                mixture = _normalize_batch_zero_one(mixture_pre)

                mix_device = mixture.to(device)
                target_device = target.to(device)
                primary_device = primary.to(device)

                if disable_model_autocast:
                    pred, _ = _sample_without_model_autocast(model, mix_device, num_steps=num_steps)
                else:
                    with torch.no_grad():
                        pred, _ = model.sample(mix_device, num_steps=num_steps, show_progress=False)

                pred = pred.unsqueeze(1).detach().cpu()
                target_cpu = target_device.detach().cpu()
                primary_cpu = primary_device.detach().cpu()

                pred_target_metrics = _compute_metric_triplet(pred, target_cpu, primary_cpu)
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
                    nmr_rows=pred_target_nmr_rows,
                )

                residual_primary = _normalize_batch_zero_one(mixture - 0.25 * pred)
                residual_primary_metrics = _compute_metric_triplet(residual_primary, primary_cpu, target_cpu)
                write_metric_block(sample_indices, "residual_primary", num_steps, residual_primary_metrics)

                if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
                    print(
                        f"[num_steps={num_steps}] processed batch {batch_idx}/{total_batches}"
                    )

    summary_rows = []
    for (condition, steps), metric_lists in sorted(summary_acc.items(), key=lambda item: (item[0][0], item[0][1])):
        row = {
            "condition": condition,
            "steps": int(steps),
            "count": len(metric_lists["sdr"]),
        }
        for metric_name in ("mse", "mae", "sdr", "sir", "sar"):
            values = np.asarray(metric_lists[metric_name], dtype=np.float64)
            finite = np.isfinite(values)
            row[f"mean_{metric_name}"] = float(np.mean(values))
            row[f"median_{metric_name}"] = float(np.median(values))
            row[f"std_{metric_name}"] = float(np.std(values))
            row[f"finite_{metric_name}_fraction"] = float(np.mean(finite))
        if nmr_enabled:
            for metric_name in nmr_fieldnames:
                values = np.asarray(metric_lists[metric_name], dtype=np.float64)
                if values.size > 0:
                    row[f"mean_{metric_name}"] = float(np.mean(values))
                else:
                    row[f"mean_{metric_name}"] = ""
        summary_rows.append(row)
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
        "device_used": str(device),
        "pairing_notes": (
            "Pairs are formed by shuffling the split once, then pairing each item "
            "with the next item in the shuffled order, wrapping at the end."
        ),
        "metric_notes": {
            "prediction_target": (
                "Prediction evaluated against the target source, with the primary source "
                "treated as the interference basis in the span-projection decomposition."
            ),
            "input_target_reference": (
                "Normalized model-input mixture evaluated directly against the target source."
            ),
            "input_primary_reference": (
                "Normalized model-input mixture evaluated directly against the primary source."
            ),
            "residual_primary": (
                "Residual primary estimate formed as normalize_0_1(mixture - 0.25 * prediction), "
                "then evaluated against the primary source with the target treated as interference."
            ),
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
        help="Optional output directory. Defaults to <work_dir>/<model_save_prefix>_full_eval_<split>.",
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
