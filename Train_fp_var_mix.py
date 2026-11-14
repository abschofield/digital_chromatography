"""Train a variable-composition IR or NMR diffusion model."""

import argparse
import csv
import importlib
import json
import os
import time
import types
from collections import deque
from typing import Any, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


def read_smiles_list(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _normalize_zero_one(tensor: torch.Tensor) -> torch.Tensor:
    value_min = tensor.min()
    value_max = tensor.max()
    denom = (value_max - value_min).clamp_min(1e-8)
    return (tensor - value_min) / denom


def _normalize_by_max(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = tensor.max().clamp_min(1e-8)
    return tensor / scale, scale


class MixtureDataset(Dataset):
    """
    On-the-fly variable-mixture dataset.
    Returns: mixture, scaled_target, scaled_primary, pure_target, alpha, alpha_prime, smiles1, smiles2.
    """

    def __init__(
        self,
        spectra: torch.Tensor,
        smiles: list[str],
        epoch_size: int,
        debug: bool = False,
        debug_max: int = 0,
        name: str = "dataset",
        mixture_cfg: Any | None = None,
    ):
        if spectra.shape[0] != len(smiles):
            raise ValueError(
                f"Spectra rows {spectra.shape[0]} != smiles {len(smiles)}"
            )
        if mixture_cfg is None:
            raise ValueError(
                "Train_fp_var_mix.py requires dataset.mixture; the fixed 3:1 regime is intentionally unsupported."
            )
        required_keys = (
            "mode",
            "alpha_min",
            "alpha_max",
            "target_output",
            "mixture_normalization",
        )
        missing = [key for key in required_keys if mixture_cfg.get(key, None) is None]
        if missing:
            raise ValueError(f"dataset.mixture is missing required keys: {missing}")
        mode = str(mixture_cfg.mode)
        if mode != "variable_alpha":
            raise ValueError(
                f"Train_fp_var_mix.py only supports dataset.mixture.mode='variable_alpha'; got {mode!r}"
            )
        target_output = str(mixture_cfg.target_output)
        if target_output != "scaled_target":
            raise ValueError(
                "Train_fp_var_mix.py only supports dataset.mixture.target_output='scaled_target'"
            )
        mixture_normalization = str(mixture_cfg.mixture_normalization)
        if mixture_normalization != "divide_by_max":
            raise ValueError(
                "Train_fp_var_mix.py only supports dataset.mixture.mixture_normalization='divide_by_max'"
            )
        self.alpha_min = float(mixture_cfg.alpha_min)
        self.alpha_max = float(mixture_cfg.alpha_max)
        if not (0.0 <= self.alpha_min <= self.alpha_max <= 0.5):
            raise ValueError(
                "Expected minority alpha bounds to satisfy 0 <= alpha_min <= alpha_max <= 0.5; "
                f"got alpha_min={self.alpha_min} alpha_max={self.alpha_max}"
            )
        self.spectra = spectra
        self.smiles = smiles
        self.num_series = spectra.shape[0]
        self.epoch_size = int(epoch_size)
        self.debug = bool(debug)
        self.debug_max = int(debug_max)
        self.debug_count = 0
        self.name = name

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx: int):
        i = torch.randint(0, self.num_series, (1,)).item()
        j = torch.randint(0, self.num_series, (1,)).item()
        pure_target = _normalize_zero_one(self.spectra[i].unsqueeze(0))
        pure_primary = _normalize_zero_one(self.spectra[j].unsqueeze(0))
        alpha = torch.empty((), dtype=pure_target.dtype).uniform_(
            self.alpha_min,
            self.alpha_max,
        )
        primary_alpha = 1.0 - alpha
        target_pre = alpha * pure_target
        primary_pre = primary_alpha * pure_primary
        mixture_pre = target_pre + primary_pre
        mixture, mixture_scale = _normalize_by_max(mixture_pre)
        scaled_target = target_pre / mixture_scale
        scaled_primary = primary_pre / mixture_scale
        alpha_prime = alpha / mixture_scale
        if self.debug and self.debug_count < self.debug_max:
            print(f"[debug][{self.name}] raw spectra")
            print(f"[debug][{self.name}] alpha={float(alpha.item()):.6f} alpha_prime={float(alpha_prime.item()):.6f}")
            print(_summarize_tensor("pure_target", pure_target))
            print(_summarize_tensor("pure_primary", pure_primary))
            print(_summarize_tensor("mixture_pre", mixture_pre))
        if self.debug and self.debug_count < self.debug_max:
            print(_summarize_tensor("mixture_norm", mixture))
            print(_summarize_tensor("scaled_target", scaled_target))
            print(_summarize_tensor("scaled_primary", scaled_primary))
            self.debug_count += 1
        return (
            mixture,
            scaled_target,
            scaled_primary,
            pure_target,
            float(alpha.item()),
            float(alpha_prime.item()),
            self.smiles[i],
            self.smiles[j],
        )


def import_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_dataloaders(cfg) -> Tuple[Any, Any, Any]:
    data_cfg = cfg.dataset
    debug_dtype = bool(cfg.train.get("debug_dtype", False))
    debug_steps = int(cfg.train.get("debug_steps", 3))

    df = pd.read_csv(
        data_cfg.data_csv,
        index_col=0,
        dtype=np.float32,
        converters={0: str},
    )
    if data_cfg.spec_len is not None and df.shape[1] != data_cfg.spec_len:
        raise ValueError(
            f"{data_cfg.data_csv}: got {df.shape[1]} cols expected {data_cfg.spec_len}"
        )

    train_smiles = read_smiles_list(data_cfg.train_smiles)
    val_smiles = read_smiles_list(data_cfg.val_smiles)
    test_smiles = read_smiles_list(data_cfg.test_smiles)

    validation_smiles = {
        "val": val_smiles,
        "test": test_smiles,
    }[data_cfg.validation_series]

    train_spectra = torch.from_numpy(df.loc[train_smiles].to_numpy(dtype=np.float32))
    val_spectra = torch.from_numpy(df.loc[validation_smiles].to_numpy(dtype=np.float32))

    if debug_dtype:
        print("[debug][data] loaded spectra from disk")
        print(_summarize_tensor("train_spectra", train_spectra))
        print(_summarize_tensor("val_spectra", val_spectra))

    train_dataset = MixtureDataset(
        train_spectra,
        train_smiles,
        data_cfg.epoch_size.train,
        debug=debug_dtype,
        debug_max=debug_steps,
        name="train_dataset",
        mixture_cfg=data_cfg.get("mixture", None),
    )
    val_dataset = MixtureDataset(
        val_spectra,
        validation_smiles,
        data_cfg.epoch_size.val,
        debug=debug_dtype,
        debug_max=debug_steps,
        name="val_dataset",
        mixture_cfg=data_cfg.get("mixture", None),
    )
    inference_dataset = MixtureDataset(
        val_spectra,
        validation_smiles,
        data_cfg.epoch_size.inference,
        debug=False,
        debug_max=0,
        name="inference_dataset",
        mixture_cfg=data_cfg.get("mixture", None),
    )

    batch_debug_counter = {"count": 0}

    def collate_batch(samples):
        mixtures, targets, primaries, pure_targets, alphas, alpha_primes, smiles1, smiles2 = zip(*samples)
        mix_t = torch.stack(mixtures, dim=0)
        target_t = torch.stack(targets, dim=0)
        primary_t = torch.stack(primaries, dim=0)
        pure_target_t = torch.stack(pure_targets, dim=0)
        batch = torch.stack([mix_t, target_t, primary_t, pure_target_t], dim=1)
        metadata = {
            "alpha": torch.tensor(alphas, dtype=torch.float32),
            "alpha_prime": torch.tensor(alpha_primes, dtype=torch.float32),
        }
        if debug_dtype and batch_debug_counter["count"] < debug_steps:
            print("[debug][batch] after collate")
            print(_summarize_tensor("batch", batch))
            print(_summarize_tensor("alpha", metadata["alpha"]))
            print(_summarize_tensor("alpha_prime", metadata["alpha_prime"]))
            batch_debug_counter["count"] += 1
        return batch, list(smiles1), list(smiles2), metadata

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_batch,
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_batch,
    )

    print(f"Train size: {len(train_dataset)} samples")
    print(f"Val size: {len(val_dataset)} samples")

    return train_loader, val_loader, inference_loader


def instantiate_model(cfg):
    model_cfg = cfg.model
    model_cls = import_class(model_cfg.class_path)
    return model_cls(**model_cfg.init_args)


def instantiate_optimizer(cfg, model):
    optimizer_cfg = cfg.model.init_args.optimizer
    instantiate_kwargs = {
        "params": filter(lambda p: p.requires_grad, model.parameters()),
    }
    if (
        "train" in cfg
        and "weight_decay" in cfg.train
        and "weight_decay" not in optimizer_cfg
    ):
        instantiate_kwargs["weight_decay"] = cfg.train.weight_decay
    return hydra.utils.instantiate(
        optimizer_cfg,
        **instantiate_kwargs,
    )


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _append_csv_row(path: str, fieldnames: list[str], row: dict[str, Any]):
    if not path:
        return
    _ensure_parent_dir(path)
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach()
        if value.numel() == 1:
            return float(value.item())
        value = value.float().mean()
        return float(value.item())
    return float(value)


def _count_nonfinite_tensor(tensor: torch.Tensor) -> dict[str, int]:
    with torch.no_grad():
        data = tensor.detach()
        return {
            "numel": int(data.numel()),
            "finite": int(torch.isfinite(data).sum().item()),
            "nan": int(torch.isnan(data).sum().item()),
            "posinf": int(torch.isposinf(data).sum().item()),
            "neginf": int(torch.isneginf(data).sum().item()),
        }


def _tensor_stats_dict(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        data = tensor.detach()
        finite = torch.isfinite(data)
        finite_count = int(finite.sum().item())
        stats = {
            f"{name}_numel": int(data.numel()),
            f"{name}_finite": finite_count,
            f"{name}_nan": int(torch.isnan(data).sum().item()),
            f"{name}_posinf": int(torch.isposinf(data).sum().item()),
            f"{name}_neginf": int(torch.isneginf(data).sum().item()),
        }
        if finite_count > 0:
            finite_data = data[finite]
            stats[f"{name}_min"] = float(finite_data.min().item())
            stats[f"{name}_max"] = float(finite_data.max().item())
            stats[f"{name}_mean"] = float(finite_data.mean().item())
        else:
            stats[f"{name}_min"] = None
            stats[f"{name}_max"] = None
            stats[f"{name}_mean"] = None
        return stats


def _batch_component_stats(batch_tensor: torch.Tensor) -> dict[str, Any]:
    stats = {}
    component_names = ["mixture", "spectrum1", "spectrum2", "pure_target"]
    for idx, name in enumerate(component_names):
        if idx < batch_tensor.shape[1]:
            stats.update(_tensor_stats_dict(name, batch_tensor[:, idx]))
    stats.update(_tensor_stats_dict("batch", batch_tensor))
    return stats


def _find_nonfinite_sample_positions(batch_tensor: torch.Tensor) -> list[int]:
    with torch.no_grad():
        flat_mask = ~torch.isfinite(batch_tensor.view(batch_tensor.shape[0], -1))
        return torch.where(flat_mask.any(dim=1))[0].cpu().tolist()


def _model_nonfinite_stats(model: torch.nn.Module) -> dict[str, Any]:
    param_tensors = 0
    param_nonfinite_tensors = 0
    param_nan = 0
    param_inf = 0
    grad_tensors = 0
    grad_nonfinite_tensors = 0
    grad_nan = 0
    grad_inf = 0
    for _, param in model.named_parameters():
        param_tensors += 1
        counts = _count_nonfinite_tensor(param.detach())
        if counts["finite"] != counts["numel"]:
            param_nonfinite_tensors += 1
        param_nan += counts["nan"]
        param_inf += counts["posinf"] + counts["neginf"]
        if param.grad is None:
            continue
        grad_tensors += 1
        grad_counts = _count_nonfinite_tensor(param.grad.detach())
        if grad_counts["finite"] != grad_counts["numel"]:
            grad_nonfinite_tensors += 1
        grad_nan += grad_counts["nan"]
        grad_inf += grad_counts["posinf"] + grad_counts["neginf"]
    return {
        "param_tensors": param_tensors,
        "param_nonfinite_tensors": param_nonfinite_tensors,
        "param_nan": param_nan,
        "param_inf": param_inf,
        "grad_tensors": grad_tensors,
        "grad_nonfinite_tensors": grad_nonfinite_tensors,
        "grad_nan": grad_nan,
        "grad_inf": grad_inf,
    }


def _optimizer_nonfinite_stats(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    total_tensors = 0
    nonfinite_tensors = 0
    nan_total = 0
    inf_total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            total_tensors += 1
            counts = _count_nonfinite_tensor(value.detach())
            if counts["finite"] != counts["numel"]:
                nonfinite_tensors += 1
            nan_total += counts["nan"]
            inf_total += counts["posinf"] + counts["neginf"]
    return {
        "optimizer_state_tensors": total_tensors,
        "optimizer_state_nonfinite_tensors": nonfinite_tensors,
        "optimizer_state_nan": nan_total,
        "optimizer_state_inf": inf_total,
    }


def _iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)


def _module_tensor_summary(tensors) -> dict[str, Any] | None:
    items = [tensor.detach() for tensor in tensors if torch.is_tensor(tensor)]
    if not items:
        return None
    abs_max = 0.0
    sq_sum = 0.0
    count = 0
    all_finite = True
    nan_total = 0
    inf_total = 0
    shapes = []
    dtypes = []
    finite_min = None
    finite_max = None
    for tensor in items:
        data = tensor.float()
        abs_max = max(abs_max, float(data.abs().max().item()))
        sq_sum += float((data * data).sum().item())
        count += int(data.numel())
        finite = torch.isfinite(data)
        all_finite = all_finite and bool(finite.all().item())
        nan_total += int(torch.isnan(data).sum().item())
        inf_total += int((torch.isposinf(data) | torch.isneginf(data)).sum().item())
        shapes.append(tuple(data.shape))
        dtypes.append(str(tensor.dtype))
        if bool(finite.any().item()):
            finite_data = data[finite]
            current_min = float(finite_data.min().item())
            current_max = float(finite_data.max().item())
            finite_min = current_min if finite_min is None else min(finite_min, current_min)
            finite_max = current_max if finite_max is None else max(finite_max, current_max)
    return {
        "num_tensors": len(items),
        "numel": count,
        "abs_max": abs_max,
        "min": finite_min,
        "max": finite_max,
        "dtype": dtypes[0] if len(set(dtypes)) == 1 else "|".join(sorted(set(dtypes))),
        "l2_norm": sq_sum ** 0.5,
        "all_finite": all_finite,
        "nan": nan_total,
        "inf": inf_total,
        "shapes": shapes[:4],
    }


def _clone_for_save(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _clone_for_save(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_clone_for_save(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_for_save(value) for value in obj)
    return obj


def _collect_nonfinite_parameters(model: torch.nn.Module) -> dict[str, Any]:
    params = {}
    grads = {}
    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            params[name] = param.detach().cpu()
        if param.grad is not None and not torch.isfinite(param.grad.detach()).all():
            grads[name] = param.grad.detach().cpu()
    return {
        "params": params,
        "grads": grads,
    }


def _collect_nonfinite_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    param_name_by_id = {id(param): name for name, param in model.named_parameters()}
    captured = {}
    for param, state in optimizer.state.items():
        state_name = param_name_by_id.get(id(param), f"param_{len(captured)}")
        bad_items = {}
        for key, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value.detach()).all():
                bad_items[key] = value.detach().cpu()
        if bad_items:
            captured[state_name] = bad_items
    return captured


def _module_activation_summary(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    summary = {
        "conditioner_forward_abs_max": None,
        "conditioner_forward_module": None,
        "diffusion_forward_abs_max": None,
        "diffusion_forward_module": None,
        "conditioner_backward_abs_max": None,
        "conditioner_backward_module": None,
        "diffusion_backward_abs_max": None,
        "diffusion_backward_module": None,
    }
    if not snapshot:
        return summary

    module_stats = snapshot.get("module_stats", {})
    for module_name, stats in module_stats.items():
        root_name = module_name.split(".", 1)[0]
        for stage in ("forward", "backward"):
            stage_stats = stats.get(stage)
            if not stage_stats:
                continue
            key_prefix = f"{root_name}_{stage}"
            value_key = f"{key_prefix}_abs_max"
            module_key = f"{key_prefix}_module"
            abs_max = stage_stats.get("abs_max")
            if abs_max is None:
                continue
            best = summary[value_key]
            if best is None or abs_max > best:
                summary[value_key] = abs_max
                summary[module_key] = module_name
    return summary


def _parameter_tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        data = tensor.detach()
        finite = torch.isfinite(data)
        finite_count = int(finite.sum().item())
        stats = {
            "dtype": str(data.dtype),
            "numel": int(data.numel()),
            "all_finite": finite_count == int(data.numel()),
            "nan": int(torch.isnan(data).sum().item()),
            "inf": int((torch.isposinf(data) | torch.isneginf(data)).sum().item()),
            "min": None,
            "max": None,
            "abs_max": None,
        }
        if finite_count > 0:
            finite_data = data[finite].float()
            stats["min"] = float(finite_data.min().item())
            stats["max"] = float(finite_data.max().item())
            stats["abs_max"] = float(finite_data.abs().max().item())
        return stats


def _append_activation_audit_rows(
    path: str,
    fieldnames: list[str],
    phase: str,
    epoch: int,
    step: int,
    global_step: int,
    snapshot: dict[str, Any] | None,
):
    if not snapshot:
        return
    for module_name, stats in snapshot.get("module_stats", {}).items():
        root_name = module_name.split(".", 1)[0]
        for stage in ("forward", "backward"):
            stage_stats = stats.get(stage)
            if not stage_stats:
                continue
            row = {
                "phase": phase,
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "module": module_name,
                "root": root_name,
                "stage": stage,
                "dtype": stage_stats.get("dtype"),
                "min": stage_stats.get("min"),
                "max": stage_stats.get("max"),
                "abs_max": stage_stats.get("abs_max"),
                "l2_norm": stage_stats.get("l2_norm"),
                "numel": stage_stats.get("numel"),
                "all_finite": stage_stats.get("all_finite"),
                "nan": stage_stats.get("nan"),
                "inf": stage_stats.get("inf"),
            }
            _append_csv_row(path, fieldnames, row)


def _append_parameter_audit_rows(
    path: str,
    fieldnames: list[str],
    phase: str,
    epoch: int,
    step: int,
    global_step: int,
    model: torch.nn.Module,
):
    for name, param in model.named_parameters():
        param_stats = _parameter_tensor_summary(param)
        _append_csv_row(
            path,
            fieldnames,
            {
                "phase": phase,
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "name": name,
                "kind": "param",
                **param_stats,
            },
        )
        if param.grad is None:
            continue
        grad_stats = _parameter_tensor_summary(param.grad)
        _append_csv_row(
            path,
            fieldnames,
            {
                "phase": phase,
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "name": name,
                "kind": "grad",
                **grad_stats,
            },
        )


class FailureCapture:
    def __init__(
        self,
        model: torch.nn.Module,
        enabled: bool = False,
        history_steps: int = 4,
        max_event_tensors: int = 8,
    ):
        self.model = model
        self.enabled = bool(enabled)
        self.history = deque(maxlen=max(1, int(history_steps)))
        self.max_event_tensors = max(1, int(max_event_tensors))
        self.current = None
        self.handles = []
        if self.enabled:
            self._register_hooks()

    def _register_hooks(self):
        for root_name in ("conditioner", "diffusion"):
            root = getattr(self.model, root_name, None)
            if root is None:
                continue
            for name, module in root.named_modules():
                if any(module.children()):
                    continue
                full_name = root_name if not name else f"{root_name}.{name}"
                self.handles.append(
                    module.register_forward_hook(self._make_forward_hook(full_name))
                )
                self.handles.append(
                    module.register_full_backward_hook(
                        self._make_backward_hook(full_name)
                    )
                )

    def _ensure_module_entry(self, module_name: str):
        if self.current is None:
            return None
        modules = self.current["module_stats"]
        entry = modules.get(module_name)
        if entry is None:
            entry = {}
            modules[module_name] = entry
        return entry

    def _record_event(self, stage: str, module_name: str, summary: dict[str, Any], tensors):
        if self.current is None:
            return
        event_key = f"{stage}_events"
        if len(self.current[event_key]) >= self.max_event_tensors:
            return
        saved = []
        for tensor in tensors:
            if torch.is_tensor(tensor):
                saved.append(tensor.detach().cpu())
            if len(saved) >= self.max_event_tensors:
                break
        self.current[event_key].append(
            {
                "module": module_name,
                "summary": summary,
                "tensors": saved,
            }
        )

    def _make_forward_hook(self, module_name: str):
        def hook(module, inputs, output):
            if self.current is None:
                return
            tensors = list(_iter_tensors(output))
            summary = _module_tensor_summary(tensors)
            if summary is None:
                return
            entry = self._ensure_module_entry(module_name)
            entry["forward"] = summary
            if not summary["all_finite"]:
                self._record_event("forward", module_name, summary, tensors)

        return hook

    def _make_backward_hook(self, module_name: str):
        def hook(module, grad_input, grad_output):
            if self.current is None:
                return
            tensors = list(_iter_tensors(grad_output)) or list(_iter_tensors(grad_input))
            summary = _module_tensor_summary(tensors)
            if summary is None:
                return
            entry = self._ensure_module_entry(module_name)
            entry["backward"] = summary
            if not summary["all_finite"]:
                self._record_event("backward", module_name, summary, tensors)

        return hook

    def start_step(self, epoch: int, step: int, global_step: int):
        if not self.enabled:
            return
        self.current = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "module_stats": {},
            "forward_events": [],
            "backward_events": [],
        }

    def finish_step(self):
        if not self.enabled or self.current is None:
            return
        self.history.append(
            {
                "epoch": self.current["epoch"],
                "step": self.current["step"],
                "global_step": self.current["global_step"],
                "module_stats": self.current["module_stats"],
            }
        )
        self.current = None

    def snapshot(self):
        return {
            "history": list(self.history),
            "current": None if self.current is None else {
                "epoch": self.current["epoch"],
                "step": self.current["step"],
                "global_step": self.current["global_step"],
                "module_stats": self.current["module_stats"],
                "forward_events": self.current["forward_events"],
                "backward_events": self.current["backward_events"],
            },
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _scan_dataframe_split(name: str, df: pd.DataFrame, smiles: list[str]) -> dict[str, Any]:
    split_df = df.loc[smiles]
    values = split_df.to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(values)
    nan_mask = np.isnan(values)
    posinf_mask = np.isposinf(values)
    neginf_mask = np.isneginf(values)
    bad_rows = np.where(~finite_mask.all(axis=1))[0]
    zero_rows = np.where(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=1) == 0.0)[0]
    summary = {
        "split": name,
        "rows": int(values.shape[0]),
        "cols": int(values.shape[1]),
        "finite": int(finite_mask.sum()),
        "nan": int(nan_mask.sum()),
        "posinf": int(posinf_mask.sum()),
        "neginf": int(neginf_mask.sum()),
        "global_min": float(np.nanmin(values)) if values.size else None,
        "global_max": float(np.nanmax(values)) if values.size else None,
        "bad_row_count": int(len(bad_rows)),
        "bad_rows_preview": split_df.index[bad_rows[:10]].tolist(),
        "all_zero_row_count": int(len(zero_rows)),
        "all_zero_rows_preview": split_df.index[zero_rows[:10]].tolist(),
    }
    return summary


def audit_dataset(cfg) -> dict[str, Any]:
    data_cfg = cfg.dataset
    df = pd.read_csv(
        data_cfg.data_csv,
        index_col=0,
        dtype=np.float32,
        converters={0: str},
    )
    train_smiles = read_smiles_list(data_cfg.train_smiles)
    val_smiles = read_smiles_list(data_cfg.val_smiles)
    test_smiles = read_smiles_list(data_cfg.test_smiles)
    audit = {
        "data_csv": data_cfg.data_csv,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "train": _scan_dataframe_split("train", df, train_smiles),
        "val": _scan_dataframe_split("val", df, val_smiles),
        "test": _scan_dataframe_split("test", df, test_smiles),
    }
    audit_path = os.path.join(
        cfg.work_dir, f"{cfg.model_save_prefix}_dataset_audit.json"
    )
    _ensure_parent_dir(audit_path)
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(f"[debug][dataset] wrote audit to {audit_path}")
    for split_name in ["train", "val", "test"]:
        info = audit[split_name]
        print(
            f"[debug][dataset] {split_name}: rows={info['rows']} cols={info['cols']} "
            f"nan={info['nan']} posinf={info['posinf']} neginf={info['neginf']} "
            f"bad_rows={info['bad_row_count']} zero_rows={info['all_zero_row_count']} "
            f"min={info['global_min']} max={info['global_max']}"
        )
        if info["bad_rows_preview"]:
            print(f"[debug][dataset] {split_name} bad_rows_preview={info['bad_rows_preview']}")
    return audit


def save_checkpoint(model, optimizer, epoch, path, extra_state=None):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)


LOSS_CURVE_METRIC_NAMES = [
    "infer_mse",
    "infer_mae",
    "infer_sdr",
    "infer_sir",
    "infer_sar",
    "infer_scale_abs_error",
    "infer_peak_abs_error",
]


def _normalize_metric_history(loaded_metrics, history_length):
    loaded_metrics = loaded_metrics or {}
    normalized = {}
    for name in LOSS_CURVE_METRIC_NAMES:
        values = list(loaded_metrics.get(name, []))[:history_length]
        if len(values) < history_length:
            values.extend([None] * (history_length - len(values)))
        normalized[name] = values
    return normalized


def save_loss_curve(
    work_dir: str,
    prefix: str,
    train_losses,
    val_losses,
    inference_metrics,
    lrs,
):
    path = os.path.join(work_dir, f"{prefix}_losses.csv")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", *LOSS_CURVE_METRIC_NAMES])
        for epoch_idx, train_loss in enumerate(train_losses, start=1):
            val_loss = val_losses[epoch_idx - 1]
            lr = lrs[epoch_idx - 1]
            metric_values = []
            for name in LOSS_CURVE_METRIC_NAMES:
                history = inference_metrics.get(name, [])
                value = history[epoch_idx - 1] if epoch_idx - 1 < len(history) else None
                metric_values.append("" if value is None else f"{value:.8f}")
            writer.writerow(
                [
                    epoch_idx,
                    f"{train_loss:.8f}",
                    "" if val_loss is None else f"{val_loss:.8f}",
                    f"{lr:.8f}",
                    *metric_values,
                ]
            )
    os.replace(tmp_path, path)


def _parse_optional_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def load_loss_curve_history(path: str, upto_epoch: int | None = None):
    train_losses = []
    val_losses = []
    lrs = []
    inference_metrics = {name: [] for name in LOSS_CURVE_METRIC_NAMES}
    if not os.path.exists(path):
        return train_losses, val_losses, inference_metrics, lrs
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch = int(row["epoch"])
            if upto_epoch is not None and epoch > upto_epoch:
                break
            train_losses.append(float(row["train_loss"]))
            val_losses.append(_parse_optional_float(row.get("val_loss")))
            lrs.append(float(row["lr"]))
            for name in inference_metrics:
                inference_metrics[name].append(_parse_optional_float(row.get(name)))
    return train_losses, val_losses, inference_metrics, lrs


def _min_finite(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return min(finite) if finite else None


def resolve_resume_state(cfg):
    train_cfg = cfg.train if "train" in cfg else OmegaConf.create({})
    resume_from_latest = bool(train_cfg.get("resume_from_latest", False))
    resume_path = train_cfg.get("resume_checkpoint", None)
    if resume_path:
        path = os.path.expanduser(str(resume_path))
    elif resume_from_latest:
        path = os.path.join(cfg.work_dir, f"{cfg.model_save_prefix}_latest.pt")
    else:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Requested resume checkpoint does not exist: {path}"
        )
    return path


def _require_inference_num_steps(cfg) -> int:
    if "train" not in cfg or "inference_num_steps" not in cfg.train:
        raise KeyError(
            "cfg.train.inference_num_steps must be set explicitly; no default num_steps is used."
        )
    num_steps = int(cfg.train.inference_num_steps)
    if num_steps <= 0:
        raise ValueError(
            f"cfg.train.inference_num_steps must be positive, got {num_steps}"
        )
    return num_steps


def _compute_mse(pred, target):
    return torch.mean((pred - target) ** 2, dim=(1, 2))


def _compute_mae(pred, target):
    return torch.mean(torch.abs(pred - target), dim=(1, 2))


def _project_onto_vector(pred, source, eps=1e-8):
    num = torch.sum(pred * source, dim=(1, 2), keepdim=True)
    den = torch.sum(source * source, dim=(1, 2), keepdim=True) + eps
    scale = num / den
    return scale * source, scale


def _project_onto_span(pred, target, interference, eps=1e-8):
    target_flat = target.reshape(target.shape[0], -1)
    interference_flat = interference.reshape(interference.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)

    aa = torch.sum(target_flat * target_flat, dim=1)
    ab = torch.sum(target_flat * interference_flat, dim=1)
    bb = torch.sum(interference_flat * interference_flat, dim=1)
    ay = torch.sum(target_flat * pred_flat, dim=1)
    by = torch.sum(interference_flat * pred_flat, dim=1)

    det = aa * bb - ab * ab
    safe_det = torch.where(det.abs() >= eps, det, torch.ones_like(det))
    coeff_target = (ay * bb - by * ab) / safe_det
    coeff_interference = (by * aa - ay * ab) / safe_det

    degenerate = det.abs() < eps
    if degenerate.any():
        fallback_target = ay / (aa + eps)
        coeff_target = torch.where(degenerate, fallback_target, coeff_target)
        coeff_interference = torch.where(
            degenerate,
            torch.zeros_like(coeff_interference),
            coeff_interference,
        )

    span_projection = (
        coeff_target.view(-1, 1, 1) * target
        + coeff_interference.view(-1, 1, 1) * interference
    )
    solution = torch.stack([coeff_target, coeff_interference], dim=1)
    return span_projection, solution


def _decompose_prediction_components(pred, target, interference, eps=1e-8):
    signal_component, signal_scale = _project_onto_vector(pred, target, eps=eps)
    span_projection, span_coeffs = _project_onto_span(
        pred,
        target,
        interference,
        eps=eps,
    )
    interference_component = span_projection - signal_component
    artifact_component = pred - span_projection
    return {
        "signal_component": signal_component,
        "interference_component": interference_component,
        "artifact_component": artifact_component,
        "signal_scale": signal_scale.view(-1),
        "span_target_coeff": span_coeffs[:, 0],
        "span_interference_coeff": span_coeffs[:, 1],
    }


def _compute_bss_metrics(pred, target, interference, eps=1e-8):
    components = _decompose_prediction_components(
        pred,
        target,
        interference,
        eps=eps,
    )
    s_target = components["signal_component"]
    s_interf = components["interference_component"]
    e_artif = components["artifact_component"]

    target_energy = torch.sum(s_target * s_target, dim=(1, 2)) + eps
    interf_energy = torch.sum(s_interf * s_interf, dim=(1, 2)) + eps
    artif_energy = torch.sum(e_artif * e_artif, dim=(1, 2)) + eps
    total_error = torch.sum((s_interf + e_artif) ** 2, dim=(1, 2)) + eps
    mix_energy = torch.sum((s_target + s_interf) ** 2, dim=(1, 2)) + eps

    sdr = 10.0 * torch.log10(target_energy / total_error)
    sir = 10.0 * torch.log10(target_energy / interf_energy)
    sar = 10.0 * torch.log10(mix_energy / artif_energy)
    return sdr, sir, sar


def _compute_sisdr(pred, target, interference, eps=1e-8):
    components = _decompose_prediction_components(
        pred,
        target,
        interference,
        eps=eps,
    )
    s_target = components["signal_component"]
    total_error = components["interference_component"] + components["artifact_component"]
    s_energy = torch.sum(s_target * s_target, dim=(1, 2)) + eps
    e_energy = torch.sum(total_error * total_error, dim=(1, 2)) + eps
    return 10.0 * torch.log10(s_energy / e_energy)


def _compute_scale_calibration_metrics(pred, target, pure_target=None, eps=1e-8):
    basis = target if pure_target is None else pure_target
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    basis_flat = basis.reshape(basis.shape[0], -1)
    denom = torch.sum(basis_flat * basis_flat, dim=1) + eps
    pred_scale = torch.sum(pred_flat * basis_flat, dim=1) / denom
    target_scale = torch.sum(target_flat * basis_flat, dim=1) / denom
    scale_abs_error = torch.abs(pred_scale - target_scale)
    pred_peak = torch.amax(pred, dim=(1, 2))
    target_peak = torch.amax(target, dim=(1, 2))
    peak_abs_error = torch.abs(pred_peak - target_peak)
    return scale_abs_error, peak_abs_error


def _compute_inference_metrics(pred, target, interference, pure_target=None):
    mse = _compute_mse(pred, target)
    mae = _compute_mae(pred, target)
    sdr, sir, sar = _compute_bss_metrics(pred, target, interference)
    scale_abs_error, peak_abs_error = _compute_scale_calibration_metrics(
        pred,
        target,
        pure_target=pure_target,
    )
    return {
        "infer_mse": mse,
        "infer_mae": mae,
        "infer_sdr": sdr,
        "infer_sir": sir,
        "infer_sar": sar,
        "infer_scale_abs_error": scale_abs_error,
        "infer_peak_abs_error": peak_abs_error,
    }


def _compute_lr(epoch, base_lr, schedule_cfg, total_epochs):
    warmup_epochs = int(schedule_cfg.get("warmup_epochs", 0))
    hold_epochs = int(schedule_cfg.get("hold_epochs", 0))
    warmup_start_lr = float(schedule_cfg.get("warmup_start_lr", 0.0))
    min_lr = float(schedule_cfg.get("min_lr", 0.0))

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)

    if epoch <= warmup_epochs + hold_epochs:
        return base_lr

    decay_epochs = max(0, total_epochs - warmup_epochs - hold_epochs)
    if decay_epochs <= 1:
        return min_lr

    decay_idx = epoch - warmup_epochs - hold_epochs - 1
    t = decay_idx / (decay_epochs - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * t))


def _summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    if tensor is None:
        return f"{name}: None"
    with torch.no_grad():
        data = tensor.detach()
        finite = torch.isfinite(data)
        finite_count = int(finite.sum().item())
        total_count = int(data.numel())
        all_finite = finite_count == total_count
        if finite_count == 0:
            return (
                f"{name}: dtype={data.dtype} shape={tuple(data.shape)} "
                f"all_finite={all_finite} finite={finite_count}/{total_count}"
            )
        finite_data = data[finite]
        return (
            f"{name}: dtype={data.dtype} shape={tuple(data.shape)} "
            f"min={float(finite_data.min()):.6g} "
            f"max={float(finite_data.max()):.6g} "
            f"mean={float(finite_data.mean()):.6g} "
            f"std={float(finite_data.std()):.6g} "
            f"all_finite={all_finite} finite={finite_count}/{total_count}"
        )


def _grad_norm(parameters) -> float:
    total = 0.0
    count = 0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.norm(grad).item())
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _print_tensor(stage: str, name: str, tensor: torch.Tensor):
    print(f"[debug][{stage}] {_summarize_tensor(name, tensor)}")


def _print_module_param_stats(model: torch.nn.Module, stage: str):
    print(f"[debug][{stage}] model parameter stats")
    dtype_counts = {}
    nonfinite_params = 0
    for name, param in model.named_parameters():
        dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + 1
        if not torch.isfinite(param.detach()).all():
            nonfinite_params += 1
            _print_tensor(f"{stage}/params_nonfinite", name, param.detach())
    print(f"[debug][{stage}] parameter dtypes={dtype_counts}")
    print(f"[debug][{stage}] nonfinite parameter tensors={nonfinite_params}")


def _print_grad_stats(model: torch.nn.Module, stage: str, max_print: int = 6):
    shown = 0
    bad = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if shown < max_print:
            _print_tensor(stage, f"grad::{name}", grad)
            shown += 1
        if not torch.isfinite(grad).all():
            bad += 1
    print(f"[debug][{stage}] grad_nonfinite_tensors={bad}")


def _print_optimizer_state_stats(optimizer: torch.optim.Optimizer, stage: str):
    dtype_counts = {}
    total_tensors = 0
    nonfinite_tensors = 0
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                total_tensors += 1
                dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
                if not torch.isfinite(value.detach()).all():
                    nonfinite_tensors += 1
                    _print_tensor(f"{stage}/optimizer_nonfinite", f"state::{key}", value)
    print(
        f"[debug][{stage}] optimizer_state tensors={total_tensors} "
        f"dtypes={dtype_counts} nonfinite={nonfinite_tensors}"
    )


def _write_failure_artifact(
    cfg,
    path: str,
    stage: str,
    epoch: int,
    step: int,
    global_step: int,
    lr: float,
    raw_loss,
    returned_loss,
    batch_tensor: torch.Tensor,
    smiles1: list[str],
    smiles2: list[str],
    batch_stats: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    failure_capture: FailureCapture | None,
):
    if not path:
        return
    payload = {
        "model_save_prefix": cfg.model_save_prefix,
        "failure_stage": stage,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "lr": lr,
        "raw_loss": _safe_float(raw_loss),
        "returned_loss": _safe_float(returned_loss),
        "batch_stats": batch_stats,
        "smiles1": list(smiles1),
        "smiles2": list(smiles2),
        "batch_tensor": batch_tensor.detach().cpu(),
        "module_capture": failure_capture.snapshot() if failure_capture is not None else None,
        "nonfinite_model_tensors": _collect_nonfinite_parameters(model),
        "nonfinite_optimizer_state": _collect_nonfinite_optimizer_state(model, optimizer),
    }
    _ensure_parent_dir(path)
    torch.save(payload, path)
    print(f"[debug][failure] wrote failure artifact to {path}")


def _extend_dim_local(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.view(*x.shape + (1,) * (dim - x.ndim))


def _install_runtime_debug_hooks(model: torch.nn.Module, cfg):
    debug_dtype = bool(cfg.train.get("debug_dtype", False))
    debug_steps = int(cfg.train.get("debug_steps", 3))
    force_float32 = bool(cfg.train.get("force_float32", False))
    disable_model_autocast = bool(
        cfg.train.get("disable_model_autocast", force_float32)
    )
    if not debug_dtype and not disable_model_autocast:
        return

    counters = {
        "step": 0,
        "diffusion": 0,
        "sample": 0,
    }

    if hasattr(model, "_step"):
        decorated = model._step
        base_fn = getattr(decorated, "__wrapped__", decorated)

        def wrapped_step(self, batch, diffusion=None):
            use_cuda_autocast = (
                self.device.type == "cuda"
                and (not force_float32)
                and (not disable_model_autocast)
            )
            with torch.autocast(device_type="cuda", enabled=use_cuda_autocast):
                batch_in = batch.float() if force_float32 else batch
                if counters["step"] < debug_steps:
                    _print_tensor("model_step", "batch_in", batch_in)
                    _print_tensor("model_step", "mix_in", batch_in[:, 0])
                    _print_tensor("model_step", "stems_in", batch_in[:, 1:])
                loss, extra = base_fn(self, batch_in, diffusion)
                if counters["step"] < debug_steps:
                    _print_tensor("model_step", "loss", loss)
                counters["step"] += 1
                return loss, extra

        model._step = types.MethodType(wrapped_step, model)

    diffusion_obj = None
    if hasattr(model, "diffusion") and hasattr(model.diffusion, "diffusion"):
        diffusion_obj = model.diffusion.diffusion

    if diffusion_obj is not None and hasattr(diffusion_obj, "forward"):
        orig_diff_forward = diffusion_obj.forward

        def wrapped_diff_forward(self, x, prior_cond=None, **kwargs):
            use_cuda_autocast = (
                x.is_cuda and (not force_float32) and (not disable_model_autocast)
            )
            with torch.autocast(device_type="cuda", enabled=use_cuda_autocast):
                x_in = x.float() if force_float32 else x
                batch_size, device = x_in.shape[0], x_in.device
                sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
                sigmas_batch = _extend_dim_local(sigmas, dim=x_in.ndim)
                noise = torch.randn_like(x_in)
                if prior_cond is not None:
                    noise = noise + prior_cond.mean(1, keepdim=True)
                    noise = noise * prior_cond.std(1, keepdim=True)
                alphas, betas = self.get_alpha_beta(sigmas_batch)
                x_noisy = alphas * x_in + betas * noise
                v_target = alphas * noise - betas * x_in
                v_pred = self.net(x_noisy, sigmas, **kwargs)
                if hasattr(self, "loss"):
                    loss = self.loss(v_pred, v_target)
                else:
                    loss = F.mse_loss(v_pred, v_target)
                if counters["diffusion"] < debug_steps:
                    _print_tensor("diffusion", "x", x_in)
                    _print_tensor("diffusion_schedule", "sigmas", sigmas)
                    _print_tensor("diffusion_schedule", "alphas", alphas)
                    _print_tensor("diffusion_schedule", "betas", betas)
                    _print_tensor("diffusion_noise", "noise", noise)
                    _print_tensor("diffusion_forward", "x_noisy", x_noisy)
                    _print_tensor("diffusion_forward", "v_target", v_target)
                    _print_tensor("diffusion_forward", "v_pred", v_pred)
                    _print_tensor("diffusion_loss", "loss", loss)
                counters["diffusion"] += 1
                return loss

        diffusion_obj.forward = types.MethodType(wrapped_diff_forward, diffusion_obj)
        diffusion_obj._debug_original_forward = orig_diff_forward

    if hasattr(model, "sample"):
        decorated_sample = model.sample
        base_sample_fn = getattr(decorated_sample, "__wrapped__", decorated_sample)

        def wrapped_sample(self, mix, *args, **kwargs):
            use_cuda_autocast = (
                mix.is_cuda and (not force_float32) and (not disable_model_autocast)
            )
            with torch.autocast(device_type="cuda", enabled=use_cuda_autocast):
                mix_in = mix.float() if force_float32 else mix
                if counters["sample"] < debug_steps:
                    _print_tensor("sample_input", "mix", mix_in)
                out = base_sample_fn(self, mix_in, *args, **kwargs)
                if counters["sample"] < debug_steps and isinstance(out, (tuple, list)) and len(out) > 0:
                    _print_tensor("sample_output", "pred", out[0])
                counters["sample"] += 1
                return out

        model.sample = types.MethodType(wrapped_sample, model)


def _move_to_device(batch_tensor: torch.Tensor, device: torch.device, force_float32: bool):
    if force_float32:
        return batch_tensor.to(device=device, dtype=torch.float32)
    return batch_tensor.to(device=device)


def _get_base_step_fn(model: torch.nn.Module):
    class_step = getattr(type(model), "_step", None)
    if class_step is None:
        return None
    return getattr(class_step, "__wrapped__", class_step)


def _run_step_with_precision(
    model: torch.nn.Module,
    batch_tensor: torch.Tensor,
    diffusion,
    autocast_enabled: bool,
):
    base_step_fn = _get_base_step_fn(model)
    if base_step_fn is None:
        return None
    use_autocast = batch_tensor.is_cuda and autocast_enabled
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=use_autocast):
            batch_in = batch_tensor if use_autocast else batch_tensor.float()
            loss, _ = base_step_fn(model, batch_in, diffusion)
    return loss.detach()


def _maybe_run_precision_probe(
    cfg,
    model: torch.nn.Module,
    batch_tensor: torch.Tensor,
    diffusion,
    phase: str,
    epoch: int,
    step: int,
    global_step: int,
    lr: float,
    raw_loss,
    probe_csv_path: str,
    probe_fieldnames: list[str],
):
    if not probe_csv_path:
        return
    if batch_tensor.device.type != "cuda":
        return
    if bool(cfg.train.get("force_float32", False)):
        return
    probe_every = int(cfg.train.get("precision_probe_every_steps", 0))
    if probe_every <= 0:
        return
    raw_isfinite = bool(torch.isfinite(raw_loss.detach()).item())
    should_probe = ((global_step % probe_every) == 0) or (not raw_isfinite)
    if not should_probe:
        return

    fp32_loss = _run_step_with_precision(
        model, batch_tensor, diffusion, autocast_enabled=False
    )
    amp_loss = _run_step_with_precision(
        model, batch_tensor, diffusion, autocast_enabled=True
    )
    fp32_value = _safe_float(fp32_loss)
    amp_value = _safe_float(amp_loss)
    abs_diff = None
    rel_diff = None
    if fp32_value is not None and amp_value is not None:
        abs_diff = abs(fp32_value - amp_value)
        rel_diff = abs_diff / max(abs(fp32_value), 1e-12)

    row = {
        "phase": phase,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "lr": lr,
        "train_raw_loss": _safe_float(raw_loss),
        "train_raw_loss_isfinite": raw_isfinite,
        "amp_probe_loss": amp_value,
        "amp_probe_isfinite": bool(torch.isfinite(amp_loss).item()) if amp_loss is not None else None,
        "fp32_probe_loss": fp32_value,
        "fp32_probe_isfinite": bool(torch.isfinite(fp32_loss).item()) if fp32_loss is not None else None,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "batch_abs_max": float(batch_tensor.detach().abs().max().item()),
        "mix_abs_max": float(batch_tensor[:, 0].detach().abs().max().item()),
        "target_abs_max": float(batch_tensor[:, 1].detach().abs().max().item()),
    }
    _append_csv_row(probe_csv_path, probe_fieldnames, row)
    print(
        f"[debug][precision_probe] phase={phase} epoch={epoch} step={step} "
        f"train_raw={row['train_raw_loss']} amp={amp_value} fp32={fp32_value} "
        f"amp_finite={row['amp_probe_isfinite']} fp32_finite={row['fp32_probe_isfinite']} "
        f"rel_diff={rel_diff}"
    )


def train(cfg):
    os.makedirs(cfg.work_dir, exist_ok=True)
    device = torch.device(cfg.device)
    model = instantiate_model(cfg).to(device)
    resume_checkpoint_path = resolve_resume_state(cfg)
    resume_state = None
    resume_epoch = 0
    start_epoch = 1
    enable_debug_artifacts = bool(cfg.train.get("enable_debug_artifacts", False))
    force_float32 = bool(cfg.train.get("force_float32", False))
    debug_dtype = bool(cfg.train.get("debug_dtype", False)) if enable_debug_artifacts else False
    debug_steps = int(cfg.train.get("debug_steps", 3)) if enable_debug_artifacts else 0
    max_train_steps = int(cfg.train.get("max_train_steps", 0))
    max_val_steps = int(cfg.train.get("max_val_steps", 0))
    max_infer_steps = int(cfg.train.get("max_infer_steps", 0))
    audit_before_train = bool(cfg.train.get("audit_dataset", True)) if enable_debug_artifacts else False
    audit_only = bool(cfg.train.get("audit_dataset_only", False)) if enable_debug_artifacts else False
    stop_on_nonfinite = bool(cfg.train.get("stop_on_nonfinite", True))
    stop_on_failure_stage = bool(cfg.train.get("stop_on_failure_stage", False))
    capture_failure_tensors = bool(cfg.train.get("capture_failure_tensors", True)) if enable_debug_artifacts else False
    capture_history_steps = int(cfg.train.get("capture_history_steps", 4))
    capture_max_event_tensors = int(cfg.train.get("capture_max_event_tensors", 8))
    activation_audit_every_steps = int(cfg.train.get("activation_audit_every_steps", 0)) if enable_debug_artifacts else 0
    parameter_audit_every_steps = int(cfg.train.get("parameter_audit_every_steps", 0)) if enable_debug_artifacts else 0
    step_csv_path = os.path.join(
        cfg.work_dir, f"{cfg.model_save_prefix}_step_debug.csv"
    ) if enable_debug_artifacts else None
    precision_probe_csv_path = os.path.join(
        cfg.work_dir, f"{cfg.model_save_prefix}_precision_probe.csv"
    ) if enable_debug_artifacts else None
    failure_artifact_path = os.path.join(
        cfg.work_dir, f"{cfg.model_save_prefix}_failure_artifact.pt"
    ) if enable_debug_artifacts else None
    activation_audit_csv_path = os.path.join(
        cfg.work_dir, f"{cfg.model_save_prefix}_activation_audit.csv"
    ) if enable_debug_artifacts else None
    parameter_audit_csv_path = os.path.join(
        cfg.work_dir, f"{cfg.model_save_prefix}_parameter_audit.csv"
    ) if enable_debug_artifacts else None
    if step_csv_path and os.path.exists(step_csv_path) and resume_checkpoint_path is None:
        os.remove(step_csv_path)
    if precision_probe_csv_path and os.path.exists(precision_probe_csv_path) and resume_checkpoint_path is None:
        os.remove(precision_probe_csv_path)
    if activation_audit_csv_path and os.path.exists(activation_audit_csv_path) and resume_checkpoint_path is None:
        os.remove(activation_audit_csv_path)
    if parameter_audit_csv_path and os.path.exists(parameter_audit_csv_path) and resume_checkpoint_path is None:
        os.remove(parameter_audit_csv_path)
    step_fieldnames = [
        "phase",
        "epoch",
        "step",
        "global_step",
        "lr",
        "raw_loss",
        "sanitized_loss",
        "returned_loss",
        "raw_loss_isfinite",
        "returned_loss_isfinite",
        "failure_stage",
        "forward_nonfinite_events",
        "backward_nonfinite_events",
        "batch_nonfinite_samples",
        "smiles1_preview",
        "smiles2_preview",
        "param_tensors",
        "param_nonfinite_tensors",
        "param_nan",
        "param_inf",
        "grad_tensors",
        "grad_nonfinite_tensors",
        "grad_nan",
        "grad_inf",
        "optimizer_state_tensors",
        "optimizer_state_nonfinite_tensors",
        "optimizer_state_nan",
        "optimizer_state_inf",
        "batch_min",
        "batch_max",
        "mixture_min",
        "mixture_max",
        "spectrum1_min",
        "spectrum1_max",
        "spectrum2_min",
        "spectrum2_max",
        "conditioner_forward_abs_max",
        "conditioner_forward_module",
        "diffusion_forward_abs_max",
        "diffusion_forward_module",
        "conditioner_backward_abs_max",
        "conditioner_backward_module",
        "diffusion_backward_abs_max",
        "diffusion_backward_module",
    ]
    precision_probe_fieldnames = [
        "phase",
        "epoch",
        "step",
        "global_step",
        "lr",
        "train_raw_loss",
        "train_raw_loss_isfinite",
        "amp_probe_loss",
        "amp_probe_isfinite",
        "fp32_probe_loss",
        "fp32_probe_isfinite",
        "abs_diff",
        "rel_diff",
        "batch_abs_max",
        "mix_abs_max",
        "target_abs_max",
    ]
    activation_audit_fieldnames = [
        "phase",
        "epoch",
        "step",
        "global_step",
        "module",
        "root",
        "stage",
        "dtype",
        "min",
        "max",
        "abs_max",
        "l2_norm",
        "numel",
        "all_finite",
        "nan",
        "inf",
    ]
    parameter_audit_fieldnames = [
        "phase",
        "epoch",
        "step",
        "global_step",
        "name",
        "kind",
        "dtype",
        "numel",
        "all_finite",
        "nan",
        "inf",
        "min",
        "max",
        "abs_max",
    ]
    disable_model_autocast = bool(
        cfg.train.get("disable_model_autocast", force_float32)
    )
    if force_float32:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        model = model.float()
        print(
            "[debug][precision] force_float32 enabled"
            + (
                ", model autocast wrappers disabled"
                if disable_model_autocast
                else ", but model autocast decorators remain active"
            )
        )
    if resume_checkpoint_path is not None:
        print(f"[resume] loading checkpoint from {resume_checkpoint_path}")
        resume_state = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(resume_state["model_state"], strict=True)
        resume_epoch = int(resume_state.get("epoch", 0))
        start_epoch = resume_epoch + 1
    print(f"Model instantiated on {device}")
    _install_runtime_debug_hooks(model, cfg)
    failure_capture = FailureCapture(
        model,
        enabled=enable_debug_artifacts and capture_failure_tensors,
        history_steps=capture_history_steps,
        max_event_tensors=capture_max_event_tensors,
    )
    if debug_dtype:
        _print_module_param_stats(model, "init")
        if hasattr(model, "diffusion") and hasattr(model.diffusion, "diffusion"):
            diffusion_obj = model.diffusion.diffusion
            if hasattr(diffusion_obj, "sigma_distribution"):
                sigmas_probe = diffusion_obj.sigma_distribution(
                    num_samples=8, device=device
                )
                _print_tensor("schedule_probe", "sigmas_probe", sigmas_probe)
                sigmas_batch = _extend_dim_local(sigmas_probe, dim=3)
                alphas_probe, betas_probe = diffusion_obj.get_alpha_beta(sigmas_batch)
                _print_tensor("schedule_probe", "alphas_probe", alphas_probe)
                _print_tensor("schedule_probe", "betas_probe", betas_probe)

    if audit_before_train:
        audit_dataset(cfg)
    if audit_only:
        print("[debug][dataset] audit_dataset_only enabled, stopping before training")
        return

    print("Starting dataloaders build")
    train_loader, val_loader, inference_loader = build_dataloaders(cfg)
    print("Dataloaders built, about to fetch batch")
    batch = next(iter(train_loader))
    batch_tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
    print("Fetched batch: ", batch_tensor.shape)
    # Loop once through training data to measure load time.
    if bool(cfg.train.get("skip_full_load_pass", True)):
        print("[debug][data] skipping full train_loader timing pass")
    else:
        load_start = time.time()
        for _ in train_loader:
            pass
        load_elapsed = time.time() - load_start
        print(f"Train data load pass time: {load_elapsed:.2f}s")
    # return # test only dataloader first

    optimizer = instantiate_optimizer(cfg, model)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        print(
            f"[resume] restored optimizer state from epoch {resume_epoch}; "
            f"next epoch will be {start_epoch}"
        )
    if debug_dtype:
        _print_optimizer_state_stats(optimizer, "optimizer_init")

    print("Starting training...")
    model.train()
    loss_csv_path = os.path.join(cfg.work_dir, f"{cfg.model_save_prefix}_losses.csv")
    train_losses = []
    val_losses = []
    lrs = []
    inference_metrics = {name: [] for name in LOSS_CURVE_METRIC_NAMES}
    if resume_state is not None:
        history = resume_state.get("history", None)
        if history is not None:
            train_losses = list(history.get("train_losses", []))[:resume_epoch]
            val_losses = list(history.get("val_losses", []))[:resume_epoch]
            lrs = list(history.get("lrs", []))[:resume_epoch]
            loaded_metrics = history.get("inference_metrics", {})
            inference_metrics = _normalize_metric_history(
                loaded_metrics,
                history_length=len(train_losses),
            )
        else:
            (
                train_losses,
                val_losses,
                inference_metrics,
                lrs,
            ) = load_loss_curve_history(loss_csv_path, upto_epoch=resume_epoch)
    schedule_cfg = cfg.train.lr_schedule if "lr_schedule" in cfg.train else None
    inference_num_steps = _require_inference_num_steps(cfg)
    best_val = float("inf")
    if resume_state is not None:
        best_val_candidates = []
        resume_best = resume_state.get("best_val", None)
        if resume_best is not None and np.isfinite(resume_best):
            best_val_candidates.append(float(resume_best))
        full_train_losses, full_val_losses, _, _ = load_loss_curve_history(loss_csv_path)
        full_best = _min_finite(full_val_losses)
        if full_best is not None:
            best_val_candidates.append(float(full_best))
        if best_val_candidates:
            best_val = min(best_val_candidates)
    debug_val_steps = int(cfg.train.get("debug_val_steps", debug_steps))
    debug_infer_steps = int(cfg.train.get("debug_infer_steps", debug_steps))
    completed_epochs = max(0, start_epoch - 1)
    global_train_step = completed_epochs * len(train_loader)
    should_stop = False
    if start_epoch > int(cfg.train.epochs):
        print(
            f"[resume] checkpoint epoch {resume_epoch} already reaches/passes configured "
            f"epochs={cfg.train.epochs}; nothing to do."
        )
        return
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        print(f"Epoch {epoch}/{cfg.train.epochs}")
        if schedule_cfg is not None:
            base_lr = optimizer.defaults["lr"]
            lr = _compute_lr(epoch, base_lr, schedule_cfg, cfg.train.epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        running_train = 0.0
        train_steps = 0
        for i, batch in enumerate(train_loader):
            smiles1 = []
            smiles2 = []
            if isinstance(batch, (tuple, list)):
                batch_tensor = batch[0]
                if len(batch) > 1:
                    smiles1 = batch[1]
                if len(batch) > 2:
                    smiles2 = batch[2]
            else:
                batch_tensor = batch
            batch_tensor = _move_to_device(batch_tensor, device, force_float32)
            batch_stats = _batch_component_stats(batch_tensor)
            bad_sample_positions = _find_nonfinite_sample_positions(batch_tensor)
            if debug_dtype and i < debug_steps:
                _print_tensor("train_batch", "batch_tensor", batch_tensor)
            optimizer.zero_grad(set_to_none=True)
            failure_capture.start_step(epoch, i, global_train_step + 1)
            raw_loss = None
            failure_stage = None
            forward_events = 0
            backward_events = 0
            backward_snapshot = None
            if hasattr(model, "_step") and hasattr(model, "augment"):
                raw_loss, _ = model._step(batch_tensor, model.diffusion)
                loss = torch.nan_to_num(raw_loss)
                if getattr(model, "diffusion_ema", None) is not None:
                    model.diffusion_ema.update()
            else:
                loss = model(batch_tensor)
                raw_loss = loss
            forward_snapshot = failure_capture.snapshot().get("current") or {}
            forward_events = len(forward_snapshot.get("forward_events", []))
            activation_summary = _module_activation_summary(forward_snapshot)
            if not torch.isfinite(raw_loss.detach()).all():
                failure_stage = "forward"
            else:
                loss.backward()
                grad_stats = _model_nonfinite_stats(model)
                backward_snapshot = failure_capture.snapshot().get("current") or {}
                backward_events = len(backward_snapshot.get("backward_events", []))
                activation_summary = _module_activation_summary(backward_snapshot)
                if grad_stats["grad_nonfinite_tensors"] > 0 or backward_events > 0:
                    failure_stage = "backward"
                else:
                    optimizer.step()
            returned_loss_value = float(loss.detach())
            raw_loss_value = _safe_float(raw_loss)
            sanitized_loss_value = _safe_float(torch.nan_to_num(raw_loss))
            running_train += returned_loss_value
            train_steps += 1
            optimizer_stats = _optimizer_nonfinite_stats(optimizer)
            model_stats = _model_nonfinite_stats(model)
            if failure_stage is None and (
                model_stats["param_nonfinite_tensors"] > 0
                or optimizer_stats["optimizer_state_nonfinite_tensors"] > 0
            ):
                failure_stage = "optimizer"
            global_train_step += 1
            should_write_activation_audit = (
                activation_audit_every_steps > 0
                and (global_train_step % activation_audit_every_steps) == 0
            ) or (failure_stage is not None)
            if should_write_activation_audit:
                snapshot_for_audit = (
                    backward_snapshot if backward_snapshot is not None and (backward_events > 0 or failure_stage == "backward")
                    else forward_snapshot
                )
                _append_activation_audit_rows(
                    activation_audit_csv_path,
                    activation_audit_fieldnames,
                    phase="train",
                    epoch=epoch,
                    step=i,
                    global_step=global_train_step,
                    snapshot=snapshot_for_audit,
                )
            should_write_parameter_audit = (
                parameter_audit_every_steps > 0
                and (global_train_step % parameter_audit_every_steps) == 0
            ) or (failure_stage is not None)
            if should_write_parameter_audit:
                _append_parameter_audit_rows(
                    parameter_audit_csv_path,
                    parameter_audit_fieldnames,
                    phase="train",
                    epoch=epoch,
                    step=i,
                    global_step=global_train_step,
                    model=model,
                )
            step_row = {
                "phase": "train",
                "epoch": epoch,
                "step": i,
                "global_step": global_train_step,
                "lr": lr,
                "raw_loss": raw_loss_value,
                "sanitized_loss": sanitized_loss_value,
                "returned_loss": returned_loss_value,
                "raw_loss_isfinite": bool(
                    torch.isfinite(raw_loss.detach()).item()
                ) if isinstance(raw_loss, torch.Tensor) and raw_loss.numel() == 1 else None,
                "returned_loss_isfinite": bool(torch.isfinite(loss.detach()).item()),
                "failure_stage": failure_stage,
                "forward_nonfinite_events": forward_events,
                "backward_nonfinite_events": backward_events,
                "batch_nonfinite_samples": len(bad_sample_positions),
                "smiles1_preview": "|".join(smiles1[:3]),
                "smiles2_preview": "|".join(smiles2[:3]),
                **model_stats,
                **optimizer_stats,
                "batch_min": batch_stats["batch_min"],
                "batch_max": batch_stats["batch_max"],
                "mixture_min": batch_stats["mixture_min"],
                "mixture_max": batch_stats["mixture_max"],
                "spectrum1_min": batch_stats["spectrum1_min"],
                "spectrum1_max": batch_stats["spectrum1_max"],
                "spectrum2_min": batch_stats["spectrum2_min"],
                "spectrum2_max": batch_stats["spectrum2_max"],
                **activation_summary,
            }
            _append_csv_row(step_csv_path, step_fieldnames, step_row)
            if failure_stage is None:
                _maybe_run_precision_probe(
                    cfg=cfg,
                    model=model,
                    batch_tensor=batch_tensor,
                    diffusion=model.diffusion,
                    phase="train",
                    epoch=epoch,
                    step=i,
                    global_step=global_train_step,
                    lr=lr,
                    raw_loss=raw_loss,
                    probe_csv_path=precision_probe_csv_path,
                    probe_fieldnames=precision_probe_fieldnames,
                )
            if i < debug_steps:
                print(
                    f"[debug][train] step={i} lr={lr:.6g} "
                    f"raw_loss={raw_loss_value:.6g} returned_loss={returned_loss_value:.6g} "
                    f"raw_isfinite={step_row['raw_loss_isfinite']} "
                    f"returned_isfinite={step_row['returned_loss_isfinite']}"
                )
                print(f"[debug][train] grad_norm={_grad_norm(model.parameters()):.6g}")
                _print_grad_stats(model, "train_grads")
                _print_optimizer_state_stats(optimizer, "train_opt_state")
            raw_loss_isfinite = step_row["raw_loss_isfinite"]
            if bad_sample_positions:
                print(
                    f"[debug][train] batch has non-finite samples at step={i}: "
                    f"indices={bad_sample_positions[:10]}"
                )
            should_handle_failure = (
                raw_loss_isfinite is False
                or model_stats["param_nonfinite_tensors"] > 0
                or (stop_on_failure_stage and failure_stage is not None)
            )
            if should_handle_failure:
                print(
                    f"[debug][train] non-finite condition at epoch={epoch} step={i} "
                    f"raw_loss={raw_loss_value} returned_loss={returned_loss_value}"
                )
                print(f"[debug][train] failure_stage={failure_stage}")
                print(f"[debug][train] batch_stats={batch_stats}")
                print(f"[debug][train] model_stats={model_stats}")
                print(f"[debug][train] optimizer_stats={optimizer_stats}")
                if capture_failure_tensors:
                    _write_failure_artifact(
                        cfg=cfg,
                        path=failure_artifact_path,
                        stage=failure_stage or "unknown",
                        epoch=epoch,
                        step=i,
                        global_step=global_train_step,
                        lr=lr,
                        raw_loss=raw_loss,
                        returned_loss=loss,
                        batch_tensor=batch_tensor,
                        smiles1=smiles1,
                        smiles2=smiles2,
                        batch_stats=batch_stats,
                        model=model,
                        optimizer=optimizer,
                        failure_capture=failure_capture,
                    )
                if stop_on_nonfinite or (stop_on_failure_stage and failure_stage is not None):
                    should_stop = True
                    failure_capture.finish_step()
                    break
            if max_train_steps > 0 and (i + 1) >= max_train_steps:
                print(f"[debug][train] stopping early at max_train_steps={max_train_steps}")
                failure_capture.finish_step()
                break
            failure_capture.finish_step()

        avg_train = running_train / max(1, train_steps)
        train_losses.append(avg_train)
        if should_stop:
            print("[debug][train] stopping after first non-finite training event")
            break

        avg_val = None
        if val_loader is not None and hasattr(model, "validation_step"):
            model.eval()
            running_val = 0.0
            val_steps = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vsmiles1 = []
                    vsmiles2 = []
                    if isinstance(vbatch, (tuple, list)):
                        vbatch_tensor = vbatch[0]
                        if len(vbatch) > 1:
                            vsmiles1 = vbatch[1]
                        if len(vbatch) > 2:
                            vsmiles2 = vbatch[2]
                    else:
                        vbatch_tensor = vbatch
                    vbatch_tensor = _move_to_device(vbatch_tensor, device, force_float32)
                    vbatch_stats = _batch_component_stats(vbatch_tensor)
                    vbad_sample_positions = _find_nonfinite_sample_positions(vbatch_tensor)
                    if debug_dtype and val_steps < debug_val_steps:
                        _print_tensor("val_batch", "vbatch_tensor", vbatch_tensor)
                    if hasattr(model, "_step"):
                        vloss, _ = model._step(vbatch_tensor, model.diffusion)
                    else:
                        vloss = model.validation_step(vbatch_tensor, 0)
                    if vloss is not None:
                        vloss_value = float(vloss.detach())
                        running_val += vloss_value
                        val_steps += 1
                        val_row = {
                            "phase": "val",
                            "epoch": epoch,
                            "step": val_steps - 1,
                            "global_step": global_train_step,
                            "lr": lr,
                            "raw_loss": vloss_value,
                            "sanitized_loss": None,
                            "returned_loss": vloss_value,
                            "raw_loss_isfinite": bool(torch.isfinite(vloss).item()),
                            "returned_loss_isfinite": bool(torch.isfinite(vloss).item()),
                            "batch_nonfinite_samples": len(vbad_sample_positions),
                            "smiles1_preview": "|".join(vsmiles1[:3]),
                            "smiles2_preview": "|".join(vsmiles2[:3]),
                            **_model_nonfinite_stats(model),
                            **_optimizer_nonfinite_stats(optimizer),
                            "batch_min": vbatch_stats["batch_min"],
                            "batch_max": vbatch_stats["batch_max"],
                            "mixture_min": vbatch_stats["mixture_min"],
                            "mixture_max": vbatch_stats["mixture_max"],
                            "spectrum1_min": vbatch_stats["spectrum1_min"],
                            "spectrum1_max": vbatch_stats["spectrum1_max"],
                            "spectrum2_min": vbatch_stats["spectrum2_min"],
                            "spectrum2_max": vbatch_stats["spectrum2_max"],
                            "conditioner_forward_abs_max": None,
                            "conditioner_forward_module": None,
                            "diffusion_forward_abs_max": None,
                            "diffusion_forward_module": None,
                            "conditioner_backward_abs_max": None,
                            "conditioner_backward_module": None,
                            "diffusion_backward_abs_max": None,
                            "diffusion_backward_module": None,
                        }
                        _append_csv_row(step_csv_path, step_fieldnames, val_row)
                        if val_steps <= debug_val_steps:
                            print(
                                f"[debug][val] step={val_steps - 1} "
                                f"lr={lr:.6g} loss={vloss_value:.6g} "
                                f"isfinite={torch.isfinite(vloss).item()}"
                            )
                        if vbad_sample_positions or not torch.isfinite(vloss):
                            print(
                                f"[debug][val] non-finite condition at epoch={epoch} "
                                f"step={val_steps - 1} bad_samples={vbad_sample_positions[:10]}"
                            )
                            print(f"[debug][val] batch_stats={vbatch_stats}")
                            if stop_on_nonfinite and (vbad_sample_positions or not torch.isfinite(vloss)):
                                should_stop = True
                                break
                        if max_val_steps > 0 and val_steps >= max_val_steps:
                            print(f"[debug][val] stopping early at max_val_steps={max_val_steps}")
                            break
            if val_steps > 0:
                avg_val = running_val / val_steps
            model.train()

        val_losses.append(avg_val)
        if should_stop:
            print("[debug][val] stopping after first non-finite validation event")
            break

        avg_infer = {name: None for name in inference_metrics}
        if inference_loader is not None and hasattr(model, "sample"):
            model.eval()
            sums = {name: 0.0 for name in inference_metrics}
            total = 0
            infer_steps = 0
            with torch.no_grad():
                for ibatch in inference_loader:
                    ismiles1 = []
                    ismiles2 = []
                    if isinstance(ibatch, (tuple, list)):
                        ibatch_tensor = ibatch[0]
                        if len(ibatch) > 1:
                            ismiles1 = ibatch[1]
                        if len(ibatch) > 2:
                            ismiles2 = ibatch[2]
                    else:
                        ibatch_tensor = ibatch
                    ibatch_tensor = _move_to_device(ibatch_tensor, device, force_float32)
                    ibatch_stats = _batch_component_stats(ibatch_tensor)
                    ibad_sample_positions = _find_nonfinite_sample_positions(ibatch_tensor)
                    mix = ibatch_tensor[:, 0]
                    target = ibatch_tensor[:, 1]
                    interference = ibatch_tensor[:, 2]
                    pure_target = ibatch_tensor[:, 3] if ibatch_tensor.shape[1] > 3 else target
                    pred, _ = model.sample(
                        mix,
                        num_steps=inference_num_steps,
                        show_progress=False,
                    )
                    pred = pred.unsqueeze(1)
                    metrics = _compute_inference_metrics(
                        pred,
                        target,
                        interference,
                        pure_target=pure_target,
                    )
                    batch_size = pred.shape[0]
                    total += batch_size
                    for name, values in metrics.items():
                        sums[name] += float(values.sum().detach())
                    infer_steps += 1
                    infer_row = {
                        "phase": "infer",
                        "epoch": epoch,
                        "step": infer_steps - 1,
                        "global_step": global_train_step,
                        "lr": lr,
                        "raw_loss": None,
                        "sanitized_loss": None,
                        "returned_loss": None,
                        "raw_loss_isfinite": None,
                        "returned_loss_isfinite": bool(torch.isfinite(pred).all().item()),
                        "batch_nonfinite_samples": len(ibad_sample_positions),
                        "smiles1_preview": "|".join(ismiles1[:3]),
                        "smiles2_preview": "|".join(ismiles2[:3]),
                        **_model_nonfinite_stats(model),
                        **_optimizer_nonfinite_stats(optimizer),
                        "batch_min": ibatch_stats["batch_min"],
                        "batch_max": ibatch_stats["batch_max"],
                        "mixture_min": ibatch_stats["mixture_min"],
                        "mixture_max": ibatch_stats["mixture_max"],
                        "spectrum1_min": ibatch_stats["spectrum1_min"],
                        "spectrum1_max": ibatch_stats["spectrum1_max"],
                        "spectrum2_min": ibatch_stats["spectrum2_min"],
                        "spectrum2_max": ibatch_stats["spectrum2_max"],
                        "conditioner_forward_abs_max": None,
                        "conditioner_forward_module": None,
                        "diffusion_forward_abs_max": None,
                        "diffusion_forward_module": None,
                        "conditioner_backward_abs_max": None,
                        "conditioner_backward_module": None,
                        "diffusion_backward_abs_max": None,
                        "diffusion_backward_module": None,
                    }
                    _append_csv_row(step_csv_path, step_fieldnames, infer_row)
                    if debug_dtype and total <= debug_infer_steps * max(1, pred.shape[0]):
                        _print_tensor("infer", "mix", mix)
                        _print_tensor("infer", "pred", pred)
                    if ibad_sample_positions or not torch.isfinite(pred).all():
                        print(
                            f"[debug][infer] non-finite condition at epoch={epoch} "
                            f"step={infer_steps - 1} bad_samples={ibad_sample_positions[:10]}"
                        )
                        print(f"[debug][infer] batch_stats={ibatch_stats}")
                        if stop_on_nonfinite:
                            should_stop = True
                            break
                    if max_infer_steps > 0 and total >= max_infer_steps * max(1, pred.shape[0]):
                        print(
                            f"[debug][infer] stopping early at approx max_infer_steps={max_infer_steps}"
                        )
                        break
            if total > 0:
                for name, total_value in sums.items():
                    avg_infer[name] = total_value / total
            model.train()

        for name in inference_metrics:
            inference_metrics[name].append(avg_infer[name])

        save_loss_curve(
            cfg.work_dir,
            cfg.model_save_prefix,
            train_losses,
            val_losses,
            inference_metrics,
            lrs,
        )

        if avg_val is not None and avg_val < best_val:
            best_val = avg_val
            best_path = os.path.join(cfg.work_dir, f"{cfg.model_save_prefix}_best.pt")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_path,
                extra_state={
                    "best_val": best_val,
                    "global_train_step": global_train_step,
                    "history": {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "lrs": lrs,
                        "inference_metrics": inference_metrics,
                    },
                },
            )

        if epoch % cfg.train.save_every == 0:
            latest_path = os.path.join(
                cfg.work_dir, f"{cfg.model_save_prefix}_latest.pt"
            )
            save_checkpoint(
                model,
                optimizer,
                epoch,
                latest_path,
                extra_state={
                    "best_val": best_val,
                    "global_train_step": global_train_step,
                    "history": {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "lrs": lrs,
                        "inference_metrics": inference_metrics,
                    },
                },
            )
        if should_stop:
            print("[debug] stopping after first non-finite inference event")
            break
    failure_capture.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/variable_ir.yaml",
        help="Path to training config file.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if "train" not in cfg:
        cfg.train = OmegaConf.create({})
    if "enable_debug_artifacts" not in cfg.train:
        cfg.train.enable_debug_artifacts = False
    if "force_float32" not in cfg.train:
        cfg.train.force_float32 = True
    if "disable_model_autocast" not in cfg.train:
        cfg.train.disable_model_autocast = True
    if "debug_dtype" not in cfg.train:
        cfg.train.debug_dtype = bool(cfg.train.enable_debug_artifacts)
    if "debug_steps" not in cfg.train:
        cfg.train.debug_steps = 5 if cfg.train.enable_debug_artifacts else 0
    if "debug_val_steps" not in cfg.train:
        cfg.train.debug_val_steps = cfg.train.debug_steps
    if "debug_infer_steps" not in cfg.train:
        cfg.train.debug_infer_steps = cfg.train.debug_steps
    if "skip_full_load_pass" not in cfg.train:
        cfg.train.skip_full_load_pass = True
    if "audit_dataset" not in cfg.train:
        cfg.train.audit_dataset = bool(cfg.train.enable_debug_artifacts)
    if "audit_dataset_only" not in cfg.train:
        cfg.train.audit_dataset_only = False
    if "capture_failure_tensors" not in cfg.train:
        cfg.train.capture_failure_tensors = bool(cfg.train.enable_debug_artifacts)
    if "stop_on_nonfinite" not in cfg.train:
        cfg.train.stop_on_nonfinite = True
    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
    model = instantiate_model(cfg)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    print("Top-level modules:")
    for name, module in model.named_children():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f" - {name}: {module.__class__.__name__} (trainable={trainable:,})")
        for subname, submodule in module.named_children():
            subtrainable = sum(
                p.numel() for p in submodule.parameters() if p.requires_grad
            )
            print(f"   - {subname}: {submodule.__class__.__name__} (trainable={subtrainable:,})")

    train(cfg)


if __name__ == "__main__":
    main()
