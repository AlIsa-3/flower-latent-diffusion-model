import pathlib
import torch
import torch.utils.data as data
import torch.optim as optim

from vae.vae_utils import EarlyStop, vae_preprocess_transforms
from utils.ImageLoader import FlattenedImageFolder
from vae.vae_architecture import VAE
from vae.VAELoss import VAELoss
from vae.train import training_loop


def main(vae_input_channels, vae_hidden_dim, vae_latent_dim, lr, loss_beta, logger):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    BATCH_SIZE = 8
    RANDOM_SEED = 423

    path = pathlib.Path("flattened_dataset")
    dataset = FlattenedImageFolder(path, transform=vae_preprocess_transforms)

    train_data, val_data, test_data = data.random_split(
        dataset, lengths=[0.8, 0.1, 0.1], generator=torch.manual_seed(RANDOM_SEED)
    )
    train_loader = data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.manual_seed(RANDOM_SEED),
    )
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE)

    model = VAE(
        input_channels=vae_input_channels,
        hidden_dimensions=vae_hidden_dim,
        latent_dimensions=vae_latent_dim,
    ).to(device)
    criterion = VAELoss(beta=loss_beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stop = EarlyStop(patience=10, min_delta=100, checkpoint_path="best_vae.ckpt")

    losses = training_loop(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=50,
        early_stop=early_stop,
        device=device,
        logger=logger,
    )
    return losses


if __name__ == "__main__":
    import logging, os
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    NUM_HIDDEN_DIM = 32
    NUM_LATENT_DIM = 8
    LR = 0.001
    LOSS_BETA = 1
    logs_filename = f"logs/{timestamp}_hidden_dim_{NUM_HIDDEN_DIM}_latent_dim_{NUM_LATENT_DIM}_lr_{LR}_beta_{LOSS_BETA}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(logs_filename)],
    )
    logger = logging.getLogger("TrainingLogger")

    losses = main(
        vae_input_channels=3,
        vae_hidden_dim=NUM_HIDDEN_DIM,
        vae_latent_dim=NUM_LATENT_DIM,
        lr=LR,
        loss_beta=LOSS_BETA,
        logger=logger,
    )

    logging.shutdown()
