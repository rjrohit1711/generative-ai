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
1. Run `cd  image_generation_ai`
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare data in a folder(Labelled data is not a requiremnt as model is trained on unlabelled data).
4. Train: `python scripts/train.py --data_dir PATH`
5. Generate: `python scripts/generate.py`

## Output 
- Model outputs a set of random images which were used in training.
  
![image](https://github.com/user-attachments/assets/945cd616-be2b-4a7f-97f9-9af3ff7df148)
