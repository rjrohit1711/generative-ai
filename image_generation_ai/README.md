# Image Generation GAN

## Overview
Modular DCGAN implementation to train on custom datasets and generate images.

## Structure
- `scripts/data_loader.py` – Data loading with `CustomImageDataset` and transforms.
- `scripts/generator.py` – `Generator` class producing 128×128 images.
- `scripts/discriminator.py` – `Discriminator` class for 128×128 inputs.
- `scripts/train.py` – Training loop, model saving.
- `scripts/generate.py` – Load generator and visualize images.
- `setup_check.py` – Environment verification script.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare data in a folder.
3. Train: `python scripts/train.py --data_dir PATH`
4. Generate: `python scripts/generate.py`
