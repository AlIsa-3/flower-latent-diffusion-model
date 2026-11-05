import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms.v2 as v2
import torch
import time
from collections import defaultdict

from vae.vae_architecture import VAE
from vae.VAELoss import VAELoss
from vae.vae_utils import EarlyStop

from typing import Literal

type LossTuple = tuple[float, float, float]
type LossHistories = dict[str, defaultdict[str, list[float]]]
type LatentStatistics = list[dict[str, float]]


def train(
    model: VAE,
    criterion: VAELoss,
    optimizer: optim.Optimizer,
    loader: data.DataLoader,
    device: Literal["cpu", "mps", "cuda"] = "cpu",
) -> LossTuple:
    model.train()
    train_loss = train_kl_divergence = train_reconstruction_loss = train_samples = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.size(0)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, kl_divergence, reconstruction_loss = criterion(outputs, batch)

        loss.backward()
        optimizer.step()

        # Add losses
        train_loss += loss.item() * batch_size
        train_kl_divergence += kl_divergence.item() * batch_size
        train_reconstruction_loss += reconstruction_loss.item() * batch_size
        train_samples += batch_size

    train_loss /= train_samples
    train_kl_divergence /= train_samples
    train_reconstruction_loss /= train_samples

    return train_loss, train_kl_divergence, train_reconstruction_loss


def validate(
    model: VAE,
    criterion: VAELoss,
    loader: data.DataLoader,
    device: Literal["cpu", "mps", "cuda"] = "cpu",
) -> tuple[LossTuple, dict[str, float]]:
    model.eval()

    val_loss = val_kl_divergence = val_reconstruction_loss = 0.0
    val_samples = 0
    epoch_mus = []
    epoch_logvars = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.size(0)

            mu, logvar = model.encode(batch)

            # Deterministic reconstruction using mu
            reconstruction = model.decode(mu)

            loss, kl_divergence, reconstruction_loss = criterion(
                (reconstruction, mu, logvar), batch
            )

            # Add losses
            val_loss += loss.item() * batch_size
            val_kl_divergence += kl_divergence.item() * batch_size
            val_reconstruction_loss += reconstruction_loss.item() * batch_size
            val_samples += batch_size

            # Keep track of each batch's mu and std to calculate latent statistics
            epoch_mus.append(mu.cpu())
            epoch_logvars.append(logvar.cpu())

        val_loss /= val_samples
        val_kl_divergence /= val_samples
        val_reconstruction_loss /= val_samples

        epoch_mus = torch.cat(epoch_mus)
        epoch_logvars = torch.cat(epoch_logvars)

        mu_mean = epoch_mus.mean().item()
        mu_std = epoch_mus.std().item()

        logvar_mean = epoch_logvars.mean().item()
        logvar_std = epoch_logvars.std().item()

        latent_statistics = {
            "mu_mean": mu_mean,
            "mu_std": mu_std,
            "logvar_mean": logvar_mean,
            "logvar_std": logvar_std,
        }

        return (val_loss, val_kl_divergence, val_reconstruction_loss), latent_statistics


def print_epoch_results(
    epoch: int,
    train_losses: LossTuple,
    validation_losses: LossTuple,
    latent_statistics: dict[str, float],
    elapsed_time: float,
) -> None:
    print(
        f"Epoch {epoch+1}: Time Elapsed = {elapsed_time:.3f}\n\n"
        f"Training: Loss = {train_losses[0]:.3f} | KL = {train_losses[1]:.3f} | Reconstruction = {train_losses[2]:.3f}\n"
        f"Validation: Loss = {validation_losses[0]:.3f} | KL = {validation_losses[1]:.3f} | Reconstruction = {validation_losses[2]:.3f}\n\n"
        f"Latent Statistics:\nMu:\nMean: {latent_statistics['mu_mean']:.4f} | std: {latent_statistics['mu_std']:.4f}\n"
        f"Logvar:\nMean: {latent_statistics['logvar_mean']:.4f} | std: {latent_statistics['logvar_std']:.4f}\n\n"
    )


def training_loop(
    model: VAE,
    criterion: VAELoss,
    optimizer: optim.Optimizer,
    train_data: data.DataLoader,
    val_data: data.DataLoader,
    num_epochs: int = 10,
    early_stop: EarlyStop = EarlyStop(),
    device: Literal["cpu", "mps", "cuda"] = "cpu",
) -> tuple[LossHistories, LatentStatistics]:
    losses: dict[str, defaultdict[str, list[float]]] = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
    }
    epoch_latent_statistics: list[dict[str, float]] = []

    for epoch in range(num_epochs):
        if device == "mps":
            torch.mps.synchronize()
        epoch_start = time.perf_counter()
        train_losses = train(model, criterion, optimizer, train_data, device)

        # Keep track of epoch training losses
        losses["train"]["loss"].append(train_losses[0])
        losses["train"]["kl_divergence"].append(train_losses[1])
        losses["train"]["reconstruction_loss"].append(train_losses[2])

        # Keep track of validation losses and statistics
        validation_losses, latent_statistics = validate(
            model, criterion, val_data, device
        )
        losses["validation"]["loss"].append(validation_losses[0])
        losses["validation"]["kl_divergence"].append(validation_losses[1])
        losses["validation"]["reconstruction_loss"].append(validation_losses[2])
        epoch_latent_statistics.append(latent_statistics)

        if device == "mps":
            torch.mps.synchronize()
        epoch_finish = time.perf_counter()

        print_epoch_results(
            epoch,
            train_losses,
            validation_losses,
            latent_statistics,
            epoch_finish - epoch_start,
        )

        # Check Early Stopping
        should_stop = early_stop(
            metric=validation_losses[0],  # Use total validation loss as the metric
            model=model,
            epoch=epoch + 1,
            optimizer=optimizer,
        )

        print(early_stop.status())

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return losses, epoch_latent_statistics


if __name__ == "__main__":
    import pathlib
    from utils.ImageLoader import FlattenedImageFolder

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    BATCH_SIZE = 8
    RANDOM_SEED = 423

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Scale to [-1,1] for Tanh activation in VAE decoder
        ]
    )

    path = pathlib.Path("flattened_dataset")
    dataset = FlattenedImageFolder(path, transform=transforms)

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

    model = VAE(input_channels=3, hidden_dimensions=32, latent_dimensions=8).to(device)
    criterion = VAELoss(beta=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    )
