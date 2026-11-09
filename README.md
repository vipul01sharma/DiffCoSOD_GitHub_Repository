# DiffCoSOD

**DiffCoSOD: Diffusion Guided Saliency and Correlation Learning for Co-Salient Object Detection**

This repository is a research implementation organized from the submitted DiffCoSOD manuscript. It implements the manuscript's four-stage pipeline:

1. **SLNet** — saliency learning with a ResNet-50 branch, compact ViT branch, and SFG-style fusion.
2. **CLNet** — saliency correlation (SCRL-style) and consensus correlation (CCRL-style) representations followed by a coarse co-saliency decoder.
3. **DRNet** — conditional mask-space diffusion refinement with a 1-channel mask target and coarse prediction `Y0` as conditioning.
4. **GADG** — group-aware fusion of saliency prior, saliency-correlation and consensus-correlation features, injected into the conditional denoiser at every reverse step.

The manuscript explicitly states that SLNet/CLNet (including SFG, SCRL and CCRL) follow Tong et al., while DRNet and GADG are the components introduced by DiffCoSOD. This repository keeps that distinction explicit in the code and documentation.

> **Important reproducibility note:** the manuscript does not contain enough implementation detail to guarantee bit-for-bit equivalence to the Tong et al. baseline. The SLNet/CLNet portion here is therefore a clean reimplementation of the described architecture, not a claim to be the authors' official Tong et al. source. Replace it with the official baseline implementation/checkpoint if exact baseline reproduction is required.

## Manuscript-aligned configuration

| Setting | Value |
|---|---:|
| Input resolution | 352 x 352 |
| Training episode | 5 images / semantic group |
| Diffusion training horizon | T = 1000 |
| Noise schedule | linear |
| DRNet channels | 32, 64, 128, 256 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 5e-4 |
| Epochs | 100 |
| Loss weights | lambda1=1.0, lambda2=1.0, lambda3=0.5, lambda4=1.0 |
| Reported hardware | NVIDIA RTX 4090 |
| Reported inference batch | 1 |

These values are taken from the submitted manuscript. The reported paper runtime is approximately 2.9/5.8/8.7/11.6 seconds for 250/500/750/1000 sampling steps on a single RTX 4090 at 352x352; the repository does **not** treat those measurements as reproduced until `benchmark_runtime.py` is run on the target hardware.

## Repository layout

```text
DiffCoSOD/
├── configs/
│   └── diffcosod.yaml
├── diffcosod/
│   ├── data/
│   │   └── dataset.py
│   ├── metrics/
│   │   └── saliency_metrics.py
│   ├── models/
│   │   ├── backbone.py
│   │   ├── saliency.py
│   │   ├── correlation.py
│   │   ├── gadg.py
│   │   ├── diffusion.py
│   │   └── model.py
│   ├── losses.py
│   └── utils.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── split_groups.py
│   └── benchmark_runtime.py
├── train.py
├── infer.py
├── evaluate.py
├── control_train.py
├── requirements.txt
├── environment.yml
├── setup.py
├── CITATION.cff
└── docs/
    ├── paper_alignment.md
    └── reviewer_control_experiment.md
```

## Installation

```bash
conda env create -f environment.yml
conda activate diffcosod
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset format

The code uses a license-safe manifest instead of redistributing benchmark datasets. Prepare the publicly available datasets according to their respective licenses and create a JSON manifest:

```json
{
  "group_0001": [
    {"image": "images/a.jpg", "mask": "masks/a.png"},
    {"image": "images/b.jpg", "mask": "masks/b.png"},
    {"image": "images/c.jpg", "mask": "masks/c.png"},
    {"image": "images/d.jpg", "mask": "masks/d.png"},
    {"image": "images/e.jpg", "mask": "masks/e.png"}
  ]
}
```

For inference, masks can be omitted. Paths may be absolute or relative to the manifest directory.

The manuscript describes training on DUTS-class and COCO-SEG and testing on CoCA, CoSOD3k and CoSal2015, with five-image training episodes sampled from one semantic group.

## Training

```bash
python train.py --config configs/diffcosod.yaml \
  --train-manifest /data/DUTS_COCO/train.json \
  --val-manifest /data/DUTS_COCO/val.json \
  --output-dir runs/diffcosod
```

For a quick smoke test:

```bash
python train.py --config configs/diffcosod.yaml \
  --train-manifest examples/train.json \
  --val-manifest examples/val.json \
  --output-dir runs/smoke \
  --epochs 1 --max-steps 2 --timesteps 20
```

## Inference

```bash
python infer.py --config configs/diffcosod.yaml \
  --manifest /data/CoCA/test.json \
  --checkpoint runs/diffcosod/best.pt \
  --output-dir predictions/coca \
  --sampling-steps 1000
```

Intermediate reverse-diffusion states can be saved with `--save-trajectory`.

## Evaluation

```bash
python evaluate.py \
  --manifest /data/CoCA/test.json \
  --pred-dir predictions/coca
```

The included evaluator reports MAE and thresholded F-measure, plus lightweight implementations of S-measure and E-measure. For a publication-grade reproduction, use the same official benchmark evaluation implementation used for the comparison table and record the exact version/commit.

## Deterministic refinement control

The reviewer-requested control is included as a separate model:

```bash
python control_train.py --config configs/diffcosod.yaml \
  --train-manifest /data/train.json \
  --output-dir runs/deterministic_control
```

The control keeps the same coarse prediction and GADG conditioning pathway but replaces stochastic diffusion sampling with recurrent deterministic refinement. **No numerical control result is hard-coded into this repository.** Results must be generated by actually training and evaluating the control.

## Runtime benchmark

```bash
python scripts/benchmark_runtime.py \
  --config configs/diffcosod.yaml \
  --checkpoint runs/diffcosod/best.pt \
  --manifest /data/CoCA/test.json \
  --sampling-steps 250 500 750 1000
```

The script writes raw timing records to JSON so the reported runtime can be independently audited.

## Reproducibility checklist

Before claiming that the repository reproduces the manuscript, record:

- Git commit hash.
- PyTorch, CUDA and GPU versions.
- Dataset release/version and manifest hash.
- Random seed.
- Checkpoint SHA-256.
- Number of sampling steps.
- Input resolution and batch size.
- Measured runtime and warm-up policy.
- Whether the official Tong et al. implementation or this reimplementation was used for SLNet/CLNet.

## Citation

```bibtex
@article{diffcosod,
  title   = {DiffCoSOD: Diffusion Guided Saliency and Correlation Learning for Co-Salient Object Detection},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

Do not fill author/venue metadata until the final accepted manuscript provides the canonical bibliographic information.
