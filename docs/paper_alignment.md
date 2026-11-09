# Paper-to-code alignment

This document records what is directly specified by the submitted manuscript and what is an implementation choice.

## Directly specified

- SLNet and CLNet, including SFG/SCRL/CCRL, follow Tong et al.
- DRNet and GADG are the proposed components.
- Coarse prediction: `Y0 = D(Fs, Fc)`.
- Diffusion target: ground-truth co-saliency mask.
- Coarse prediction `Y0` is conditioning input to the denoiser.
- T=1000 and a linear noise schedule.
- Conditional U-Net channels 32/64/128/256.
- GADG combines saliency, saliency-correlation and consensus-correlation information and conditions the denoiser at each step.
- Training: AdamW, lr 1e-4, weight decay 5e-4, 100 epochs.
- Loss weights: 1.0, 1.0, 0.5, 1.0.
- Five-image semantic-group episodes.
- Evaluation datasets: CoCA, CoSOD3k, CoSal2015.

## Implementation choices that should be audited before publication

- Exact SFG/SCRL/CCRL internals are not fully specified in the DiffCoSOD manuscript; this repository provides a compact reimplementation.
- Exact pretrained ViT initialization is not specified beyond a 384-dimensional, 6-block, 6-head compact ViT description.
- Exact data augmentation details are not fully enumerated.
- Exact DDPM variance parameterization/sampling equation is implemented in the conventional DDPM form.
- The lightweight evaluator is not intended to replace official CoSOD evaluation code.

## Reviewer-control experiment

`control_train.py` provides a deterministic recurrent refinement model using the same coarse prediction and GADG feature pathway. It is intentionally included without pre-filled metrics. Run it, save raw outputs and timing, and only then add numerical results to the manuscript/rebuttal.
