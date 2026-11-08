import pathlib
import logging
import torch
import torch.utils.data as data
import torch.optim as optim

from vae.vae_utils import EarlyStop, vae_preprocess_transforms, setup_logger
from utils.ImageLoader import FlattenedImageFolder
from utils import utils
from vae.vae_architecture import VAE
from vae.VAELoss import VAELoss
from vae.train import training_loop


def main():
    device = utils.device_select()

    # Training Params
    NUM_EPOCHS = 500

    # Random Generators
    RANDOM_SEED = 423
    torch.manual_seed(RANDOM_SEED)
    split_generator = torch.Generator().manual_seed(325)
    loader_generator = torch.Generator().manual_seed(342)

    # VAE Params
    NUM_INPUT_CHANNELS = 3
    NUM_HIDDEN_DIM = 32
    NUM_LATENT_DIM = 8

    # Optimizer Params
    LR = 0.001

    # Loss Function Params
    LOSS_BETA = 1

    # Early Stop Params
    CHECKPOINT_PATH = pathlib.Path("checkpoints/best_vae.ckpt")
    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_MIN_DELTA = 100

    BATCH_SIZE = 8

    logger = setup_logger(NUM_HIDDEN_DIM, NUM_LATENT_DIM, LR, LOSS_BETA)
    path = pathlib.Path("flattened_dataset")
    dataset = FlattenedImageFolder(path, transform=vae_preprocess_transforms)

    train_data, val_data, test_data = data.random_split(
        dataset, lengths=[0.8, 0.1, 0.1], generator=split_generator
    )
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
    criterion = VAELoss(beta=LOSS_BETA).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    early_stop = EarlyStop(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        checkpoint_path=CHECKPOINT_PATH,
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
