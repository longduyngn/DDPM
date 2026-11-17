# DDPM from Scratch (PyTorch)

A from-scratch implementation of **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) for
MNIST and CIFAR-10.

## Features

- **U-Net** with residual blocks, GroupNorm, multi-head self-attention (`scaled_dot_product_attention`)
  and sinusoidal time embeddings — `src/network.py`, `src/modules.py`
- Noise schedules: `linear`, `cosine` and others — `DDPM_Scheduler` in `src/modules.py`
- Training: **automatic mixed precision** (autocast + GradScaler), **multi-GPU `DataParallel`**,
  **EMA** weights (timm `ModelEmaV3`), resumable checkpoints, fixed seeds, loss curve export
- Inference: reverse diffusion from pure noise with the EMA model, saving the denoising sequence

## Structure

```
src/config.py    dataset, hyperparameters, U-Net channels/attention layout, paths
src/modules.py   time embeddings, ResBlock, Attention, DDPM scheduler, visualization
src/network.py   U-Net
train.py         training loop
inference.py     sampling from a checkpoint
notebook/        exploratory notebook
```

## Usage

```bash
pip install torch torchvision timm einops tqdm matplotlib

# choose DATASET = 'MNIST' or 'CIFAR10' in src/config.py
python train.py          # checkpoints/ , report/loss_plot.png
python inference.py      # report/generated_sequence_*.png
```

Default settings: 1000 diffusion steps, cosine schedule, batch size 64, lr 2e-4, EMA decay 0.9999.
