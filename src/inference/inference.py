from vae.vae_utils import unNormalize
from sampling.sampler_utils import sampling_loop
from sampling.samplers import euler_ancestral_update
from utils.visualization import image_to_uint8
from PIL import Image
import torch
import pathlib
import time
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
        debug: bool = False,
        save_image: bool = True,
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
        if debug:
            timesteps = (
                torch.linspace(
                    starting_timestep - 1, 0, steps=steps, device=self.device
                )
                .round()
                .long()
            )
        else:
            timesteps = torch.tensor(
                sorted(random.sample(range(starting_timestep), k=steps), reverse=True),
                device=self.device,
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
        if save_image:
            self._save_image(decoded, name=image_name, ext=image_ext)
        return decoded

    def _save_image(
        self,
        image: torch.Tensor,
        name: str,
        ext: str = "jpg",
    ) -> None:
        save_path = pathlib.Path(f"{name}.{ext}")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensor to numpy image
        numpy_image = image_to_uint8(image)

        # Save using Pillow
        pil_image = Image.fromarray(numpy_image)
        pil_image.save(save_path)
