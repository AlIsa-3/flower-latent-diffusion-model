import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch

import logging
import time
from typing import Literal

from utils import utils
from utils.logger import get_null_logger
from vae.vae_architecture import VAE

from unet.unet_architecture import UNet
from unet.diffusion_utils import CosineNoiseScheduler

type TrainingLosses = tuple[list[float], list[float]]


@torch.no_grad()
def encode_images(
    images: torch.Tensor, vae: VAE, scaling_factor: torch.Tensor
) -> torch.Tensor:
    mu, logvar = vae.encode(images)
    z = vae.reparameterize(mu, logvar)

    return z * scaling_factor


@torch.no_grad()
def prepare_validation_data(
    vae: VAE,
    scaling_factor: torch.Tensor,
    scheduler: CosineNoiseScheduler,
    val_loader: data.DataLoader,
    device: Literal["cuda", "mps", "cpu"],
    batch_size: int = 8,
) -> data.DataLoader:
    all_latents = []
    all_noise = []
    all_timesteps = []

    vae.eval()

    if len(val_loader) == 0:
        raise ValueError("Validation Loader cannot be empty")

    for batch in val_loader:
        batch = batch.to(device)
        current_batch_size = batch.size(0)

        latents = encode_images(batch, vae, scaling_factor)

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            low=0,
            high=scheduler.num_timesteps,
            size=(current_batch_size,),
            device=device,
        )

        all_latents.append(latents.cpu())
        all_noise.append(noise.cpu())
        all_timesteps.append(timesteps.cpu())

    validation_dataset = data.TensorDataset(
        torch.cat(all_latents), torch.cat(all_noise), torch.cat(all_timesteps)
    )
    return data.DataLoader(validation_dataset, batch_size=batch_size)


def noise_latents(
    latents: torch.Tensor, time_steps: torch.Tensor, scheduler: CosineNoiseScheduler
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(latents)
    return scheduler.add_noise(latents, noise, time_steps), noise


def train(
    unet: UNet,
    vae: VAE,
    scheduler: CosineNoiseScheduler,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: data.DataLoader,
    device: Literal["cuda", "mps", "cpu"],
) -> dict[str, float]:
    if len(train_loader) == 0:
        raise ValueError("Empty dataloader received")

    unet.train()

    training_loss = 0
    grad_norms = []
    num_samples = 0

    predictions_sum = 0
    predictions_square_sum = 0
    prediction_count = 0

    noise_sum = 0
    noise_square_sum = 0
    noise_count = 0

    for batch in train_loader:
        batch = batch.to(device)
        batch_size = batch.size(0)
        optimizer.zero_grad(set_to_none=True)
        # Encode Images
        latents = encode_images(batch, vae, unet.latent_scaling_factor)

        # Add Noise
        time_steps = torch.randint(
            low=0, high=scheduler.num_timesteps, size=(batch_size,), device=device
        )
        noised_latents, noise = noise_latents(latents, time_steps, scheduler)

        predictions = unet(noised_latents, time_steps)

        # compare to actual noise added
        loss = criterion(predictions, noise)
        loss.backward()
        training_loss += loss.item() * batch_size
        num_samples += batch_size

        # Add to statistics
        predictions = predictions.detach().float()
        predictions_sum += predictions.sum().item()
        predictions_square_sum += predictions.square().sum().item()
        prediction_count += predictions.numel()

        noise = noise.detach().float()
        noise_sum += noise.sum().item()
        noise_square_sum += noise.square().sum().item()
        noise_count += noise.numel()

        # clip gradient norm to prevent instability
        grad_norm = nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())

        optimizer.step()

    # Training Statistics
    training_loss /= num_samples
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    max_grad_norm = max(grad_norms)
    prediction_mean = predictions_sum / prediction_count
    noise_mean = noise_sum / noise_count
    clipped_batches = sum(norm > 1.0 for norm in grad_norms)
    clipping_percentage = 100 * clipped_batches / len(grad_norms)
    return {
        "loss": training_loss,
        "average_pre_clip_grad_norm": avg_grad_norm,
        "max_pre_clip_grad_norm": max_grad_norm,
        "prediction_mean": prediction_mean,
        "prediction_std": max(
            ((predictions_square_sum / prediction_count) - prediction_mean**2), 0
        )
        ** 0.5,
        "noise_mean": noise_mean,
        "noise_std": max(((noise_square_sum / noise_count) - noise_mean**2), 0) ** 0.5,
        "clipping_percentage": clipping_percentage,
    }


def validate(
    unet: UNet,
    scheduler: CosineNoiseScheduler,
    criterion: nn.Module,
    val_loader: data.DataLoader,
    device: Literal["cuda", "mps", "cpu"],
) -> float:
    val_loss = 0
    num_samples = 0

    unet.eval()
    with torch.inference_mode():
        for latents, noises, timesteps in val_loader:
            latents = latents.to(device)
            noises = noises.to(device)
            timesteps = timesteps.to(device)

            current_batch_size = latents.size(0)

            noised_latents = scheduler.add_noise(latents, noises, timesteps)

            predictions = unet(noised_latents, timesteps)
            loss = criterion(predictions, noises)
            val_loss += loss.item() * current_batch_size
            num_samples += current_batch_size

    if num_samples == 0:
        raise ValueError("Cannot divide by 0, Empty dataloader")
    val_loss /= num_samples
    return val_loss


def training_loop(
    unet: UNet,
    vae: VAE,
    scheduler: CosineNoiseScheduler,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    num_epochs: int,
    early_stop: utils.EarlyStop | None = None,
    device: Literal["cuda", "mps", "cpu"] = "cpu",
    logger: logging.Logger | None = None,
) -> TrainingLosses:
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    if logger is None:
        logger = get_null_logger()

    if early_stop is None:
        early_stop = utils.EarlyStop()

    val_loader = prepare_validation_data(
        vae, unet.latent_scaling_factor, scheduler, val_loader, device
    )
    epoch_training_losses = []
    epoch_val_losses = []
    logger.info(f"Beginning Training with {num_epochs} epochs")
    logger.info(
        f"Early Stopping Initialized: min_delta {early_stop.min_delta} | patience {early_stop.patience}"
    )
    for epoch in range(num_epochs):
        if device == "mps":
            torch.mps.synchronize()
        epoch_start = time.perf_counter()
        train_statistics = train(
            unet, vae, scheduler, criterion, optimizer, train_loader, device
        )

        val_loss = validate(unet, scheduler, criterion, val_loader, device)

        epoch_training_losses.append(train_statistics["loss"])
        epoch_val_losses.append(val_loss)

        # Statistics
        if device == "mps":
            torch.mps.synchronize()
        epoch_finish = time.perf_counter()

        log_statistics(
            logger, epoch, train_statistics, val_loss, epoch_finish - epoch_start
        )

        should_stop = early_stop(
            metric=val_loss,
            model=unet,
            epoch=epoch + 1,
            optimizer=optimizer,
        )

        logger.info(f"Early stopping status: {early_stop.status()}")

        if should_stop:
            logger.info("Early stopping triggered.")
            logger.info(f"Saving best model to {early_stop.checkpoint_path}")
            logger.info(f"Saving latest model to {early_stop.latest_checkpoint_path}")
            break

    return epoch_training_losses, epoch_val_losses


def log_statistics(
    logger: logging.Logger,
    epoch: int,
    train_statistics: dict[str, float],
    val_loss: float,
    epoch_time: float,
) -> None:
    train_statistics_str = "\n".join(
        [f"{key} = {value:.4f}" for key, value in train_statistics.items()]
    )
    logger.info(
        f"\nEpoch {epoch + 1} | Time Taken: {epoch_time:.3f}s\n\n"
        f"{train_statistics_str}\n"
        f"Validation Loss: {val_loss:.4f}"
    )
