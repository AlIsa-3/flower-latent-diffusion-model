from unet.unet_architecture import UNet
from unet.diffusion_utils import CosineNoiseScheduler
from vae.vae_architecture import VAE
from inference.inference import Inference
from utils.utils import device_select
import torch.utils.data as data
from torchmetrics.image.kid import KernelInceptionDistance
import torch
from typing import Literal

from metrics.metrics_utils import (
    generate_unet_test_set,
    load_real_test_set,
    test_transforms,
)


def calculate_kid(
    real: data.DataLoader,
    generated: data.DataLoader,
    device: Literal["cuda", "mps", "cpu"] = "cpu",
) -> tuple[float, float]:
    kid = KernelInceptionDistance(subset_size=50, normalize=True)
    for images in real:
        kid.update(images.to(device), real=True)

    for images in generated:
        kid.update(images.to(device), real=False)

    kid_mean, kid_std = kid.compute()
    return kid_mean.item(), kid_std.item()


def calculate_vae_kid(
    real: data.DataLoader, vae: VAE, device: Literal["cuda", "mps", "cpu"] = "cpu"
) -> tuple[float, float]:
    vae = vae.to(device)
    vae.eval()

    kid = KernelInceptionDistance(subset_size=50, normalize=True)
    for images in real:
        kid.update(images.to(device), real=True)

    # Use mu to for deterministic reconstructions
    for images in real:
        images = images.to(device)
        mu, logvar = vae.encode(images)
        reconstruction = vae.decode(mu)
        kid.update(reconstruction.to(device), real=False)

    kid_mean, kid_std = kid.compute()
    return kid_mean.item(), kid_std.item()


def main():
    # V-Prediction Score
    device = device_select()
    vae = VAE(3, 32, 16)
    unet = UNet(16, 32, 256)
    pred_type = "v"
    sch = CosineNoiseScheduler(1000, zero_terminal_snr=pred_type == "v")
    vae.load_state_dict(torch.load("models/vae/best_vae_256.pth"))
    unet.load_state_dict(torch.load("models/unet/best_unet_v_256.pth"))
    inference = Inference(unet, vae, sch, device, pred_type)
    generated_loader = generate_unet_test_set(
        inference=inference,
        count=818,
        prediction_type=pred_type,
        batch_size=8,
        transforms=test_transforms,
        regenerate=True,
    )
    test_loader = load_real_test_set(
        generator_seed=423, batch_size=8, transforms=test_transforms
    )
    print("Calculating v-prediction KID Score")
    kid_score_v = calculate_kid(test_loader, generated_loader, device="cpu")

    # Epsilon Prediction Score
    pred_type = "eps"
    unet.load_state_dict(torch.load("models/unet/best_unet_eps_256.pth"))
    sch = CosineNoiseScheduler(1000, zero_terminal_snr=pred_type == "v")
    inference = Inference(unet, vae, sch, device, pred_type)
    generated_loader = generate_unet_test_set(
        inference=inference,
        count=818,
        prediction_type=pred_type,
        batch_size=8,
        transforms=test_transforms,
        regenerate=True,
    )
    print("Calculating Epsilon Prediction KID Score")
    kid_score_eps = calculate_kid(test_loader, generated_loader, device="cpu")

    # VAE KID Score
    print("Calculating VAE KID Score")
    test_loader = load_real_test_set(
        generator_seed=325, batch_size=8, transforms=test_transforms
    )
    kid_score_vae = calculate_vae_kid(test_loader, vae, device="cpu")

    print("Saving Scores to Disk")
    with open("KID_Scores.txt", "w") as file:
        file.write(f"v-pred U-Net KID = {kid_score_v[0]} +- {kid_score_v[1]}\n")
        file.write(f"eps pred U-Net KID = {kid_score_eps[0]} +- {kid_score_eps[1]}\n")
        file.write(f"VAE KID = {kid_score_vae[0]} +- {kid_score_vae[1]}")


if __name__ == "__main__":
    main()
