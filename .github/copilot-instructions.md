# Copilot instructions for Hybrid-Video-Synth-TransGan

## Build, test, and lint commands

This repository is currently research-oriented and does **not** define a formal build system, lint config, or automated test suite (no `pyproject.toml`, `setup.py`, `pytest.ini`, or `tests/` discovered).

| Purpose | Command | Notes |
| --- | --- | --- |
| Main training run (CLI) | `python trash/train.py --data-dir "EchoNet-Dynamic/sub all frames"` | Uses defaults for image size, batch size, epochs, checkpoints, and samples. |
| Quick smoke run (short sanity check) | `python trash/train.py --data-dir "EchoNet-Dynamic/sub all frames" --epochs 1 --save-every 1 --batch-size 2 --num-workers 0` | Fastest practical execution check in this repo. |
| Single test | Not available | No automated test framework is configured. |
| Lint | Not available | No linter configuration is present. |

## High-level architecture

This project trains a hybrid VQGAN + PatchGAN pipeline for synthetic echocardiogram frame generation.

1. **Data assets** live in `EchoNet-Dynamic/` (CSV metadata plus video/frame folders).
2. **Preprocessing utilities** (`extract_frames_from_avi`, `process_avi_folder`, `consolidate_images_from_subfolders`) are defined in `test.ipynb` and mirrored in `trash/train.py` to convert `.avi` videos into flattened image folders.
3. **Training entry points**:
   - `test.ipynb`: interactive experimentation and training flow.
   - `trash/train.py`: scriptable training entry point with `argparse`.
   - `trash/run_train.ipynb`: notebook wrapper that launches `train.py` as a subprocess (documented there as a macOS workaround for `num_workers > 0`).
4. **Model stack** in `trash/train.py`:
   - `VQGAN` encoder/quantizer/decoder autoencoder
   - `PatchGANDiscriminator`
   - Composite training loss = L1 reconstruction + LPIPS perceptual + VQ codebook + adversarial terms
5. **Artifacts** are written to:
   - `checkpoints_proid/` (`vqgan_heart_ep{epoch}.pth`)
   - `samples/` (generated sample frames)

## Key conventions in this codebase

1. Training data is treated as **single-channel grayscale** and normalized to `[-1, 1]`; decoder output uses `tanh`, and sample visualization converts back with `(x_hat + 1) / 2`.
2. Default training dataset path is `"EchoNet-Dynamic/sub all frames"` (note the space in the directory name; keep it quoted in CLI commands).
3. Device selection convention is `mps` first, then `cuda`, then `cpu` unless `--device` is explicitly passed.
4. macOS multiprocessing compatibility is handled explicitly (`torch.multiprocessing.set_start_method("spawn", force=True)` in script; notebook flow highlights protected execution patterns).
5. Checkpoint payload convention stores epoch plus model/discriminator/optimizer state dicts under fixed keys (`model_state_dict`, `disc_state_dict`, `opt_ae_state_dict`, `opt_disc_state_dict`).
6. Comments and logs are mixed English/Spanish; preserve that style when extending nearby code.
