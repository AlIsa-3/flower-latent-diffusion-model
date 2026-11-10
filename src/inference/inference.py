from vae.vae_utils import unNormalize
from sampling.sampler_utils import sampling_loop
from sampling.samplers import euler_ancestral_update

import torch
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
import random


class Inference:
    def __init__(
        self,
        unet,
        vae,
        scheduler,
        device,
        prediction_type: Literal["eps", "v"],
    ):
        if prediction_type not in ("eps", "v"):
            raise ValueError(
                f"Prediction type must be either 'eps' or 'v', received {prediction_type}"
            )

        self.vae = vae.to(device=device)
        self.vae.eval()
        self.unet = unet.to(device=device)
        self.unet.eval()
        self.scheduler = scheduler.to(device=device)
        self.device = device
        self.prediction_type: Literal["eps", "v"] = prediction_type
        self.training_timesteps = scheduler.num_timesteps

    @torch.inference_mode()
    def sample(
        self,
        steps: int,
        seed: int = -1,
        sampler=euler_ancestral_update,
        latent_size: tuple[int, int, int] = (8, 32, 32),
        image_name: str = "image",
        image_ext: str = "jpg",
        starting_timestep: int = 900,
    ) -> torch.Tensor:
        if steps < 1 or steps > self.training_timesteps:
            raise ValueError(f"Steps must be between 1 and {self.training_timesteps}")
        if starting_timestep > self.training_timesteps or starting_timestep < 1:
            raise ValueError(
                f"starting_timestep must be between 1 and {self.training_timesteps}"
            )
        if starting_timestep - steps < 0:
            raise ValueError("starting_timestep must be greater than or equal to steps")
        if seed < -1:
            raise ValueError("seed must be greater than 0 | or -1 to use a random seed")

        if seed == -1:
            seed = time.time_ns()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        timesteps = torch.tensor(
            sorted(random.sample(range(900), k=25), reverse=True), device=self.device
        )
        sigmas = self.scheduler.sigmas[timesteps]
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        latent = torch.randn(
            latent_size, generator=generator, device=self.device
        ).unsqueeze(0)

        denoised_latent = sampling_loop(
            self.unet,
            sigmas,
            timesteps,
            latent,
            sampler,
            self.device,
            self.prediction_type,
        )
        decoded = self.vae.decode(denoised_latent)
        self._save_image(decoded, name=image_name, ext=image_ext)
        return decoded

    def _save_image(self, image: torch.Tensor, name: str, ext: str = "jpg") -> None:
        save_path = pathlib.Path(f"{name}.{ext}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
        ax.imshow(
            unNormalize(
                np.transpose(image.detach().cpu().squeeze(0), (1, 2, 0)), 0.5, 0.5
            )
        )
        ax.axis("off")
        fig.savefig(str(save_path), bbox_inches="tight")
        plt.close()
