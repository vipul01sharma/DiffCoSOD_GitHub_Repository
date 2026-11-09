# Parameter-matched deterministic refinement control

## Purpose

Test whether the gain attributed to diffusion is explained primarily by additional parameters or iterative refinement capacity.

## Protocol

1. Use the same train/validation/test manifests.
2. Keep image resolution, group size, preprocessing and seed fixed.
3. Keep the adopted saliency/correlation trunk fixed.
4. Use the same `Y0` and GADG representation.
5. Replace stochastic reverse diffusion with deterministic recurrent refinement.
6. Match trainable parameter count as closely as practical to the DRNet branch and report the exact count.
7. Report Emax, S-measure, Fmax and MAE on CoCA.
8. Report inference time and peak GPU memory under the same hardware/input/batch settings.

## Do not fabricate the control result

The manuscript currently reports diffusion ablations but not a verified parameter-matched deterministic result. The repository therefore contains the experiment implementation but no invented numerical outcome.
