import torch.utils.data as data
import torch.optim as optim
import torch
import time
import logging
from collections import defaultdict

from vae.vae_architecture import VAE
from vae.VAELoss import VAELoss
from vae.vae_utils import EarlyStop, kl_per_channel
from utils.logger import get_null_logger

from typing import Literal

type LossTuple = tuple[float, float, float]
type LossHistories = dict[str, defaultdict[str, list[float]]]
type LatentStatistics = dict[str, float | list[float]]


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
) -> tuple[LossTuple, LatentStatistics]:
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

        # Calculate per-channel statistics  (only want the channels)
        mu_channel_mean = epoch_mus.mean(dim=(0, 2, 3))
        mu_channel_std = epoch_mus.std(dim=(0, 2, 3))

        logvar_channel_mean = epoch_logvars.mean(dim=(0, 2, 3))
        logvar_channel_std = epoch_logvars.std(dim=(0, 2, 3))

        channel_kls = kl_per_channel(epoch_mus, epoch_logvars)

        latent_statistics = {
            "mu_mean": mu_mean,
            "mu_std": mu_std,
            "logvar_mean": logvar_mean,
            "logvar_std": logvar_std,
            "mu_channel_mean": mu_channel_mean.tolist(),
            "mu_channel_std": mu_channel_std.tolist(),
            "logvar_channel_mean": logvar_channel_mean.tolist(),
            "logvar_channel_std": logvar_channel_std.tolist(),
            "kl_per_channel": channel_kls.tolist(),
        }

    return (val_loss, val_kl_divergence, val_reconstruction_loss), latent_statistics


def log_epoch_results(
    epoch: int,
    train_losses: LossTuple,
    validation_losses: LossTuple,
    latent_statistics: LatentStatistics,
    elapsed_time: float,
    logger: logging.Logger,
) -> None:
    logger.info(
        f"\nEpoch {epoch+1}: Time Elapsed = {elapsed_time:.3f}\n\n"
        f"Training: Loss = {train_losses[0]:.3f} | KL = {train_losses[1]:.3f} | Reconstruction = {train_losses[2]:.3f}\n"
        f"Validation: Loss = {validation_losses[0]:.3f} | KL = {validation_losses[1]:.3f} | Reconstruction = {validation_losses[2]:.3f}\n"
        f"KL Divergence Per Channel: {latent_statistics['kl_per_channel']}\n"
        f"Latent Statistics:\nMu:\n Mean = {latent_statistics['mu_mean']:.4f} | std = {latent_statistics['mu_std']:.4f}\n"
        f"Channel Mu:\nMean = {latent_statistics['mu_channel_mean']}\nstd = {latent_statistics['mu_channel_std']}\n"
        f"Logvar:\n Mean = {latent_statistics['logvar_mean']:.4f} | std = {latent_statistics['logvar_std']:.4f}\n"
        f"Channel Logvar:\nMean = {latent_statistics['logvar_channel_mean']}\nstd = {latent_statistics['logvar_channel_std']}\n\n"
    )


def training_loop(
    model: VAE,
    criterion: VAELoss,
    optimizer: optim.Optimizer,
    train_data: data.DataLoader,
    val_data: data.DataLoader,
    num_epochs: int = 10,
    early_stop: EarlyStop | None = None,
    device: Literal["cpu", "mps", "cuda"] = "cpu",
    logger: logging.Logger | None = None,
) -> tuple[LossHistories, list[LatentStatistics]]:
    if logger is None:
        logger = get_null_logger()

    if early_stop is None:
        early_stop = EarlyStop()

    losses: dict[str, defaultdict[str, list[float]]] = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
    }
    epoch_latent_statistics: list[LatentStatistics] = []

    logger.info(f"Beginning Training with {num_epochs} Epochs")
    logger.info(
        f"Early Stopping Criteria: Patience = {early_stop.patience} | Minimum Delta = {early_stop.min_delta}"
    )
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}")
        if device == "mps":
            torch.mps.synchronize()
        epoch_start = time.perf_counter()

        logger.info("Training Start")
        train_losses = train(model, criterion, optimizer, train_data, device)
        logger.info("Training Finished")

        # Keep track of epoch training losses
        losses["train"]["loss"].append(train_losses[0])
        losses["train"]["kl_divergence"].append(train_losses[1])
        losses["train"]["reconstruction_loss"].append(train_losses[2])

        # Keep track of validation losses and statistics
        logger.info("Validation Start")
        validation_losses, latent_statistics = validate(
            model, criterion, val_data, device
        )
        logger.info("Validation Finished")
        losses["validation"]["loss"].append(validation_losses[0])
        losses["validation"]["kl_divergence"].append(validation_losses[1])
        losses["validation"]["reconstruction_loss"].append(validation_losses[2])
        epoch_latent_statistics.append(latent_statistics)

        if device == "mps":
            torch.mps.synchronize()
        epoch_finish = time.perf_counter()

        log_epoch_results(
            epoch,
            train_losses,
            validation_losses,
            latent_statistics,
            epoch_finish - epoch_start,
            logger,
        )

        # Check Early Stopping
        should_stop = early_stop(
            metric=validation_losses[0],  # Use total validation loss as the metric
            model=model,
            epoch=epoch + 1,
            optimizer=optimizer,
        )

        logger.info(early_stop.status())

        if should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    logger.info("Finished Model Training")
    return losses, epoch_latent_statistics
