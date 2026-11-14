"""Initialize experimental-IR fine-tuning from the fixed-IR checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    source_path = Path(cfg.finetuning.source_checkpoint_path)
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(source_path, map_location="cpu")
    checkpoint["epoch"] = 0
    checkpoint["best_val"] = None
    checkpoint["model_save_prefix"] = str(cfg.model_save_prefix)
    checkpoint["history"] = {
        "train_losses": [],
        "val_losses": [],
        "lrs": [],
        "inference_metrics": {},
    }

    init_path = work_dir / f"{cfg.model_save_prefix}_init.pt"
    latest_path = work_dir / f"{cfg.model_save_prefix}_latest.pt"
    if init_path.exists() or latest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing fine-tuning checkpoint in "
            f"{work_dir}"
        )
    torch.save(checkpoint, init_path)
    torch.save(checkpoint, latest_path)
    print(f"Initialized fine-tuning checkpoint: {latest_path}")


if __name__ == "__main__":
    main()
