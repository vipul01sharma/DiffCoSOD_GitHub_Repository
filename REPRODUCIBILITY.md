# Reproducibility Guide

This guide explains how to reproduce the results, predictions and runtime measurements of **DiffCoSOD** and how to verify the released files.

## 1. What can be reproduced

| Claim | How to reproduce | Needed |
|---|---|---|
| Co-saliency metrics (Emax, S-measure, Fmax, MAE) on CoCA, CoSOD3k, CoSal2015 | Sections 5 and 6 | Checkpoint or your own training run |
| Ablation results | Re-train with the configs in `configs/` and evaluate as in Section 6 | Datasets, GPU |
| Inference cost (latency and memory versus sampling steps) | Section 7 | Checkpoint, GPU |
| Parameter-matched deterministic control | `control_train.py` (see `docs/reviewer_control_experiment.md`) | No result is reported for this experiment |

## 2. Released files

The following files are attached to the GitHub Release for this code version, together with a `SHA256SUMS` file:

- `best.pt`: the checkpoint used for the reported results
- `predictions_coca.zip`, `predictions_cosod3k.zip`, `predictions_cosal2015.zip`: raw prediction maps (PNG), one per input image, with the same file names as the dataset images
- `manifests.zip`: group, image and mask lists and the train/validation split (file names only, no images)
- `runtime.json` and `runtime_raw.csv`: the runtime and memory measurements described in Section 7

To verify downloaded files, place them next to `SHA256SUMS` and run:

```bash
sha256sum -c SHA256SUMS
```

## 3. Environment

```bash
conda env create -f environment.yml
conda activate diffcosod
pip install -e .
```

The reference hardware for the reported timings is a single NVIDIA RTX 4090. The exact GPU, driver, CUDA, PyTorch and git commit used for any measurement are recorded automatically in `runtime.json`. Record your own package versions with `pip freeze`.

All scripts take the random seed from `configs/diffcosod.yaml` (`seed: 42`). Bitwise determinism on GPU is not guaranteed, so expect small run-to-run differences.

## 4. Data preparation

Datasets are not redistributed. Obtain them under their own licenses:

- Training: DUTS-class and COCO-SEG
- Testing: CoCA, CoSOD3k and CoSal2015

Build a group manifest from a CSV with the columns `group,image,mask`, then split it:

```bash
python scripts/prepare_dataset.py --csv data/train.csv --output manifests/train_all.json
python scripts/split_groups.py --input manifests/train_all.json \
    --train manifests/train.json --val manifests/val.json --val-fraction 0.1
python scripts/prepare_dataset.py --csv data/coca.csv --output manifests/coca.json
```

## 5. Training and inference

```bash
python train.py --config configs/diffcosod.yaml \
    --train-manifest manifests/train.json --val-manifest manifests/val.json \
    --output-dir runs/diffcosod

python infer.py --config configs/diffcosod.yaml --manifest manifests/coca.json \
    --checkpoint runs/diffcosod/best.pt --output-dir predictions/coca \
    --sampling-steps 1000
```

A quick end-to-end check that needs no real data:

```bash
python scripts/create_dummy_dataset.py --output examples/data
python train.py --config configs/diffcosod.yaml \
    --train-manifest examples/data/manifest.json --val-manifest examples/data/manifest.json \
    --output-dir runs/smoke --epochs 1 --max-steps 2 --timesteps 20
```

## 6. Evaluation

```bash
python evaluate.py --manifest manifests/coca.json --pred-dir predictions/coca
```

`evaluate.py` is a lightweight implementation of MAE, F-measure, S-measure and E-measure for convenience. Results for publication should be computed with the official co-saliency benchmark evaluation code, and the evaluator used should be stated next to any table.

## 7. Runtime and memory protocol

**What is timed:** one call to `model.predict()` on one group of five 352 x 352 images, covering the encoders, the correlation and guidance modules, and the full reverse-diffusion sampling. Data loading is excluded.

**Units:** latency is reported per group and per image (per group divided by five). Per-group times should not be compared with per-image figures from other methods.

```bash
python scripts/benchmark_runtime.py --config configs/diffcosod.yaml \
    --checkpoint runs/diffcosod/best.pt --manifest manifests/coca.json \
    --sampling-steps 250 500 750 1000 --repeats 30 --warmup 5 \
    --output-dir results/runtime
```

Outputs:

- `runtime.json`: environment (GPU, driver, CUDA, PyTorch, git commit, SHA-256 of the checkpoint, config and manifest) and, for each setting, mean, standard deviation, median, 5th and 95th percentiles, per-image time and peak GPU memory.
- `runtime_raw.csv`: every individual timing.

Peak GPU memory is reset for each sampling-step setting. Report the median with the 5th to 95th percentile range and the hardware line from `runtime.json`.

## 8. Known limitations

- **Reimplementation.** SLNet and CLNet (SFG, SCRL, CCRL) are a compact reimplementation of the architecture described in the cited prior work. They are not that work's official code. Exact baseline reproduction requires its official implementation and checkpoint. See `docs/paper_alignment.md`.
- **Unspecified details.** The pretrained ViT initialization and some augmentation details are not specified in the manuscript, so results from retraining may differ from the released checkpoint.
- **Sampling with fewer than 1000 steps.** The sampler visits a subset of the 1000 training timesteps and applies the single-step DDPM update at each. It is not a re-spaced schedule, so results at fewer than 1000 steps should be read with this in mind.
- **Hardware dependence.** Latency and memory depend on the GPU, driver and PyTorch build. Compare only measurements recorded with their `runtime.json`.
- **Control experiment.** The parameter-matched deterministic control is implemented, but no result is reported for it.

## 9. Citation

Please cite the work using `CITATION.cff`.

## 10. License

The code is released under the MIT license (see `LICENSE`). Datasets must be obtained under their own licenses.
