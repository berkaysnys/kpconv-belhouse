import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.BelHouse3D import BelHouse3DSemSegDataset
from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN

# ------------------------------------------
# Configuration for BelHouse3D training
# ------------------------------------------
class BelHouse3DConfig(Config):
    dataset = 'BelHouse3D'
    task = 'cloud_segmentation'
    num_classes = 19
    input_threads = 4

    architecture = [
        'simple',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb_deformable',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary'
    ]

    # KPConv Parameters
    num_kernel_points = 15
    in_radius = 1.0
    first_subsampling_dl = 0.05
    conv_radius = 2.5
    deform_radius = 5.0
    KP_extent = 1.2
    KP_influence = 'linear'
    aggregation_mode = 'sum'
    in_features_dim = 1
    modulated = False

    # === COLAB-Friendly Training Parameters ===
    max_epoch = 25                     # ↓ like PointNet++
    batch_num = 48                     # ↑ matches your previous batch_size
    learning_rate = 0.001
    optimizer = 'adam'
    momentum = 0.98                    # unused with Adam, fine to keep
    lr_decays = {10: 0.7, 20: 0.7}     # ⬅ like step_size=10, lr_decay=0.7
    grad_clip_norm = 100.0

    # Reducing per-epoch load
    epoch_steps = 200                 # ↓ fewer steps per epoch than default (300)
    validation_size = 20             # ↓ fewer val batches
    checkpoint_gap = 5               # Save every 5 epochs

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, True]
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    segloss_balance = 'none'

    saving = True
    saving_path = 'results/BelHouse3D'  # Set to 'results/BelHouse3D' if desired


# ------------------------------------------
# Main Training Script
# ------------------------------------------
if __name__ == '__main__':
    # Load config
    config = BelHouse3DConfig()

    print("Preparing datasets...")
    train_dataset = BelHouse3DSemSegDataset(
        root='/content/data/belhouse3d/processed/semseg/IID-nonoccluded',
        split='train',
        num_points=2048
    )

    val_dataset = BelHouse3DSemSegDataset(
        root='/content/data/belhouse3d/processed/semseg/IID-nonoccluded',
        split='val',
        num_points=2048
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_num,
        shuffle=True,
        num_workers=config.input_threads,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_num,
        shuffle=False,
        num_workers=config.input_threads,
        drop_last=True
    )

    # Initialize KPConv model
    print("Initializing KPConv model...")
    model = KPFCNN(
        config=config,
        label_values=list(range(config.num_classes)),
        ignored_labels=[]
    )

    # Create trainer and train
    print("Starting training...")
    trainer = ModelTrainer(model, config)
    trainer.train(model, train_loader, val_loader, config)

    print("Training finished.")
