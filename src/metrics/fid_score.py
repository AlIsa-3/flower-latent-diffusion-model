from unet.unet_architecture import UNet
from unet.diffusion_utils import CosineNoiseScheduler
from vae.vae_architecture import VAE
from inference.inference import Inference
from utils.utils import device_select
import torch.utils.data as data
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from typing import Literal

from metrics.metrics_utils import (
    generate_unet_test_set,
    load_real_test_set,
    test_transforms,
)


def calculate_fid(
    real: data.DataLoader,
    generated: data.DataLoader,
    device: Literal["cuda", "mps", "cpu"] = "cpu",
) -> float:
    fid = FrechetInceptionDistance(normalize=True)
    for images in real:
        fid.update(images.to(device), real=True)

    for images in generated:
        fid.update(images.to(device), real=False)

    score = fid.compute()
    return score.item()


def calculate_vae_fid(
    real: data.DataLoader, vae: VAE, device: Literal["cuda", "mps", "cpu"] = "cpu"
) -> float:
    vae = vae.to(device)
    vae.eval()

    fid = FrechetInceptionDistance(normalize=True)
    for images in real:
        fid.update(images.to(device), real=True)

    # Use mu to for deterministic reconstructions
    for images in real:
        images = images.to(device)
        mu, logvar = vae.encode(images)
        reconstruction = vae.decode(mu)
        fid.update(reconstruction.to(device), real=False)

    score = fid.compute()
    return score.item()


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
    print("Calculating v-prediction FID Score")
    fid_score_v = calculate_fid(test_loader, generated_loader, device="cpu")

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
    print("Calculating Epsilon Prediction FID Score")
    fid_score_eps = calculate_fid(test_loader, generated_loader, device="cpu")

    # VAE FID Score
    print("Calculating VAE FID Score")
    test_loader = load_real_test_set(
        generator_seed=325, batch_size=8, transforms=test_transforms
    )
    fid_score_vae = calculate_vae_fid(test_loader, vae, device="cpu")

    print("Saving Scores to Disk")
    with open("FID_Scores.txt", "w") as file:
        file.write(f"v-pred U-Net FID = {fid_score_v}\n")
        file.write(f"eps pred U-Net FID = {fid_score_eps}\n")
        file.write(f"VAE FID = {fid_score_vae}")


if __name__ == "__main__":
    main()
