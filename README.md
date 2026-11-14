# Digital Chromatography

Code for training and evaluating conditional diffusion models that extract a
target IR or proton NMR spectrum from an additive spectral mixture. This
repository accompanies the preprint *Digital Chromatography: Computational
Separation of Mixed Spectra through Conditional Diffusion*.

**Interactive results explorer:** https://abschofield.github.io/digital_chromatography/  
A single self-contained page (`docs/index.html`) for browsing test examples and results; it works offline once downloaded.

**Status:** pre-release snapshot. The scripts are being checked against the reported results, and the processed data and
model checkpoints will be released with the paper.

The models support two mixture regimes:

- **Fixed composition:** 3:1 primary-to-target mixtures.
- **Variable composition:** `(1 - alpha):alpha` primary-to-target mixtures,
  trained with `alpha` between 0 and 0.4.

Processed data, split files, and model checkpoints are distributed separately
through the companion archive cited by the preprint.

## Installation

Python 3.10 and a CUDA-capable GPU are recommended.

```bash
conda create -n digital-chromatography python=3.10
conda activate digital-chromatography
pip install -r requirements.txt
pip install -e .
```

## Configuration

Select the configuration for the desired model:

| Configuration | Model |
| --- | --- |
| `configs/fixed_ir.yaml` | Fixed-composition simulated IR |
| `configs/fixed_nmr.yaml` | Fixed-composition simulated proton NMR |
| `configs/variable_ir.yaml` | Variable-composition simulated IR |
| `configs/variable_nmr.yaml` | Variable-composition simulated proton NMR |
| `configs/experimental_ir.yaml` | Fixed-composition experimental IR |

Before running a command, edit the selected configuration to set:

1. `dataset.data_csv` and the three split-file paths.
2. `evaluation.checkpoint_path` for evaluation.
3. `evaluation.split` to either `val` or `test`.
4. The desired output directory.

The spectral CSV uses canonical SMILES strings as row labels and spectral
intensities as columns. Split files contain one SMILES string per line. IR
spectra contain 900 values and proton NMR spectra contain 993 values.

## Evaluation

Evaluate a fixed-composition or experimental-IR model:

```bash
python evaluate_true_fp32_full.py --config configs/fixed_ir.yaml
python evaluate_true_fp32_full.py --config configs/fixed_nmr.yaml
python evaluate_true_fp32_full.py --config configs/experimental_ir.yaml
```

Evaluate a variable-composition model at `evaluation.minority_alpha`:

```bash
python evaluate_true_fp_var_mix_full.py --config configs/variable_ir.yaml
python evaluate_true_fp_var_mix_full.py --config configs/variable_nmr.yaml
```

Evaluate a variable-composition model across an alpha range:

```bash
python evaluate_true_fp_var_mix_alpha_sweep.py --config configs/variable_ir.yaml
python evaluate_true_fp_var_mix_alpha_sweep.py --config configs/variable_nmr.yaml
```

Add `--save-predictions` to an alpha-sweep command to retain the prediction
arrays. Run the variable-IR case studies with:

```bash
python run_var_mix_case_studies.py --config configs/variable_ir.yaml
```

Evaluation output includes per-sample metrics, summary metrics, sample-pair
assignments, and a manifest containing the checkpoint and evaluation settings.
Use `python SCRIPT.py --help` to view command-line overrides.

## Training

Train a fixed-composition model:

```bash
python Train_fp32.py --config configs/fixed_ir.yaml
python Train_fp32.py --config configs/fixed_nmr.yaml
```

Train a variable-composition model:

```bash
python Train_fp_var_mix.py --config configs/variable_ir.yaml
python Train_fp_var_mix.py --config configs/variable_nmr.yaml
```

For experimental-IR fine-tuning, set
`finetuning.source_checkpoint_path` in `configs/experimental_ir.yaml`, then run:

```bash
python prepare_experimental_finetuning.py --config configs/experimental_ir.yaml
python Train_fp32.py --config configs/experimental_ir.yaml
```

Checkpoints are written under `work_dir`. Set
`train.resume_from_latest: true` to continue an interrupted training run.

## Metrics

Evaluation reports scale-invariant SDR, MAE, and MSE. Proton NMR evaluation
also reports peak precision, recall, and false-positive rate.

For NMR peak metrics, predictions are clipped at zero and scaled to unit
maximum; targets are independently normalized to `[0, 1]`. Peaks are detected
with height 0.01 and prominence 0.03. Adjacent peaks are grouped when the
intervening valley remains above 0.1 times the smaller peak height. Matching
uses two-index padding around each peak-group interval, and true-positive
credit requires a predicted intensity of at least 50% of the target peak
intensity.

## Citation

Please cite the associated preprint and archived software release when using
this code. Citation metadata are provided in `CITATION.cff`.
