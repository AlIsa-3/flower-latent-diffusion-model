import pathlib
import logging
import time
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from vae.vae_architecture import VAE
from vae.vae_utils import vae_preprocess_transforms, calc_scaling_factor
from utils import utils
from utils.logger import setup_logger
from utils.ImageLoader import FlattenedImageFolder
from unet.unet_architecture import UNet
from unet.diffusion_utils import CosineNoiseScheduler
from unet.train import training_loop
from typing import Literal


def main():
    setup_start = time.perf_counter()
    # Choose prediction type (eps vs v)
    PREDICTION_TARGET: Literal["eps", "v"] = "v"

    SEED = 235
    torch.manual_seed(SEED)
    NUM_EPOCHS = 2000
    BATCH_SIZE = 8
    LR = 1e-4

    VAE_INPUT_CHANNELS = 3
    VAE_HIDDEN_DIMS = 32
    VAE_LATENT_DIMS = 8

    data_path = pathlib.Path("flattened_dataset/")
    vae_path = pathlib.Path("models/vae/best_vae.ckpt")

    device = utils.device_select()

    split_generator = torch.Generator().manual_seed(423)
    loader_generator = torch.Generator().manual_seed(342)

    # Load Data
    dataset = FlattenedImageFolder(data_path, transform=vae_preprocess_transforms)
    train_set, val_set, test_set = data.random_split(
        dataset, [0.8, 0.1, 0.1], generator=split_generator
    )

    train_loader = data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, generator=loader_generator
    )
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE)

    # Load and Freeze VAE
    vae = VAE(
        input_channels=VAE_INPUT_CHANNELS,
        hidden_dimensions=VAE_HIDDEN_DIMS,
        latent_dimensions=VAE_LATENT_DIMS,
    ).to(device)
    vae.load_state_dict(
        torch.load(vae_path, map_location=device, weights_only=True)["model_state_dict"]
    )
    vae.requires_grad_(False)
    vae.eval()

    # Instantiate U-Net
    UNET_HIDDEN_DIMS = 16
    UNET_TIMESTEP_EMBEDDING_DIMS = 256
    LATENT_SCALING_FACTOR = torch.tensor(
        calc_scaling_factor(vae, train_loader, device=device)
    )

    logs_filename = f"U-Net_HiddenDims_{UNET_HIDDEN_DIMS}_LatentInputDims_{VAE_LATENT_DIMS}_LR_{LR}_TimeEmbeddingDims_{UNET_TIMESTEP_EMBEDDING_DIMS}_{PREDICTION_TARGET}_prediction"
    logger = setup_logger(filename=logs_filename)
    if PREDICTION_TARGET == "eps":
        logger.info("Running training with epsilon prediction")
    elif PREDICTION_TARGET == "v":
        logger.info("Running training with v prediction")
    setup_finish = time.perf_counter()
    logger.info(f"Setup Time Taken: {setup_finish-setup_start:.5f}s")
    logger.info(
        f"Training Set Size: {len(train_set)} | Validation Set Size: {len(val_set)} | Batch Size {BATCH_SIZE}"
    )
    logger.info(
        f"U-Net info: Latent Input Dimensions = {VAE_LATENT_DIMS} | Initial Hidden Dimensions = {UNET_HIDDEN_DIMS} | Time-Step Embedding Dimensions = {UNET_TIMESTEP_EMBEDDING_DIMS}"
    )
    logger.info(f"Learning rate: {LR}")
    logger.info(f"VAE Checkpoint: {vae_path}")
    logger.info(
        f"VAE Info: VAE Input Channels = {VAE_INPUT_CHANNELS} | VAE Initial Hidden Dimensions = {VAE_HIDDEN_DIMS} | VAE Encoder Output Latent Channels = {VAE_LATENT_DIMS}"
    )
    logger.info(f"Latent Scaling Factor: {LATENT_SCALING_FACTOR:.5f}")

    unet = UNet(
        input_latent_channels=VAE_LATENT_DIMS,
        hidden_dimensions=UNET_HIDDEN_DIMS,
        timestep_embedding_dims=UNET_TIMESTEP_EMBEDDING_DIMS,
        latent_scaling_factor=LATENT_SCALING_FACTOR,
    ).to(device)

    # Instantiate Noise Scheduler
    NUM_TIMESTEPS = 1000
    scheduler = CosineNoiseScheduler(
        num_timesteps=NUM_TIMESTEPS, s=0.008, zero_terminal_snr=True
    ).to(device)
    logger.info(
        f"Noise Scheduler: {scheduler._get_name()} | Num Training Time-Steps: {NUM_TIMESTEPS} | s = {scheduler.s} | ZSNR: {scheduler.zero_terminal_snr}"
    )

    # Instantiate Optimizer and Criterion
    criterion = nn.MSELoss().to(device)
    optimizer = optim.AdamW(unet.parameters(), lr=LR, weight_decay=1e-4)
    early_stop = utils.EarlyStop(
        patience=25,
        min_delta=0.001,
        checkpoint_path=f"models/unet/best_unet_{PREDICTION_TARGET}.ckpt",
        latest_checkpoint_path=f"models/unet/latest_unet_{PREDICTION_TARGET}.ckpt",
    )

    training_loop(
        unet,
        vae,
        scheduler,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        early_stop=early_stop,
        device=device,
        logger=logger,
        prediction_type=PREDICTION_TARGET,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        logging.shutdown()
