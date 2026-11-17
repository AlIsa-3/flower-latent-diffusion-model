import pathlib
import logging
import torch
import torch.utils.data as data
import torch.optim as optim

from vae.vae_utils import vae_preprocess_transforms
from utils.ImageLoader import FlattenedImageFolder
from utils import utils
from utils.logger import setup_logger
from vae.vae_architecture import VAE
from vae.VAELoss import VAELoss
from vae.train import training_loop


def main():
    device = utils.device_select()

    # Training Params
    NUM_EPOCHS = 500
    # Assumes Square Images
    IMAGE_RESOLUTION = 256

    # Random Generators
    RANDOM_SEED = 423
    torch.manual_seed(RANDOM_SEED)
    split_generator = torch.Generator().manual_seed(325)
    loader_generator = torch.Generator().manual_seed(342)

    # VAE Params
    NUM_INPUT_CHANNELS = 3
    NUM_HIDDEN_DIM = 32
    NUM_LATENT_DIM = 16

    # Optimizer Params
    LR = 0.001

    # Loss Function Params
    LOSS_BETA = 1

    # Early Stop Params
    CHECKPOINT_PATH = pathlib.Path(f"models/vae/best_vae_{IMAGE_RESOLUTION}.ckpt")
    LATEST_CHECKPOINT_PATH = pathlib.Path(
        f"models/vae/latest_vae_{IMAGE_RESOLUTION}.ckpt"
    )
    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_MIN_DELTA = 100

    BATCH_SIZE = 16
    logger = setup_logger(
        f"VAE_hidden_dim_{NUM_HIDDEN_DIM}_latent_dim_{NUM_LATENT_DIM}_lr_{LR}_beta_{LOSS_BETA}_resolution_{IMAGE_RESOLUTION}",
        stdout=True,
    )

    path = pathlib.Path("flattened_dataset")
    dataset = FlattenedImageFolder(path, transform=vae_preprocess_transforms)

    train_data, val_data, test_data = data.random_split(
        dataset, lengths=[0.8, 0.1, 0.1], generator=split_generator
    )
    logger.info(f"Training with square images with size {IMAGE_RESOLUTION}")
    logger.info(f"Train Size: {len(train_data)} | Val Size {len(val_data)}")
    train_loader = data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE)

    model = VAE(
        input_channels=NUM_INPUT_CHANNELS,
        hidden_dimensions=NUM_HIDDEN_DIM,
        latent_dimensions=NUM_LATENT_DIM,
    ).to(device)
    criterion = VAELoss(beta=LOSS_BETA, reconstruction_loss="mae").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    early_stop = utils.EarlyStop(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        checkpoint_path=CHECKPOINT_PATH,
        latest_checkpoint_path=LATEST_CHECKPOINT_PATH,
    )

    losses = training_loop(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        early_stop=early_stop,
        device=device,
        logger=logger,
    )
    return losses


if __name__ == "__main__":
    try:
        losses = main()
    finally:
        logging.shutdown()
