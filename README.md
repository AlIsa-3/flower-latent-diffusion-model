# Flower Image Generation Via Latent Diffusion Model

<div align='justify'>
Latent diffusion models (LDMs) were implemented in PyTorch and trained on the Oxford Flowers 102 dataset. Multiple variants were trained, each with the goal of improving a shortcoming of the previous model. Below is an overview of the process, including the generated outputs of the individual models. 
</div>

## Variational Autoencoder (VAE)

<div align='justify'>
Initially a rather shallow VAE was trained at 256x256 to experiment with the architecture. The model architecture was iteratively experimented with, with the aim of investigating whether increasing the model capacity would improve the generation quality, while still remaining relatively inexpensive to use. All hidden dimension values within the model are derived as multiples of the initial hidden dimension count passed when constructing the VAE class. 
</div>

Initial VAE Configuration:
| | Initial Hidden Dimensions | Latent Dimensions|
|:---:|:---:|:---:|
|256x256| 16 | 8|
|512x512| 64 | 32|

<div align='justify'>
Both VAE models are convolutional VAEs containing residual blocks and using SiLU activation. They both have a downscaling factor of 8 (so a 256x256 image gets compressed to a 32x32 latent), however their channel counts and latent representation are different. 
Adding spatial self-attention was something which was experimented with in the VAE, both within the encoder and the decoder near the bottleneck; however, the reconstruction quality did not have enough of a meaningful improvement to continue using it. In addition, due to hardware limitations, the computational cost of adding attention meant that the spatial compression factor had to be increased to 16x causing the model to lose more spatial representation. 

The 512x512 VAE was trained to determine if having a larger image would allow the model to create more detailed flower images. Were the diffusion models trained in pixel-space, this increase may have been too much for the hardware to handle, however, by training in latent-space this increase was only from 32x32 to 64x64, which could still be computed in a reasonable amount of time.
</div>

### 256x256 VAE

<div align='justify'>
For the VAE trained at 256x256 px resolution, the model struggled to reconstruct the finer details of the flower images (such as the flower centers), however it was able to reconstruct the large-scale structures quite well. It was originally trained with mean square error (MSE), however that tended to produce overly blurry reconstructions. To try and combat this, the model was retrained using mean absolute error, with the goal of preserving the finer visual details.

Below is a sample of its performance across different epochs.
</div>
<img src=training_images/vae/32x32_32hidden_8latent.webp width=1000>

<div align='justify'>

The top row is the input image, and the bottom row is the reconstruction. Each column represents every 10th epoch (1, 11, 21, etc).

One potential cause of the difficulty in reconstruction was the size of the model. This VAE's latent representation for the 256x256 image had a dimensionality of $(8,32,32)$, which meant that it may not have had enough capacity to represent the information needed.
</div>

### 512x512 VAE
<div align='justify'>

For the 512x512 VAE, the model was trained using MSE, unlike in the 256x256 VAE, however it had the same model architecture. In order to try and mitigate the issues of not having enough model capacity, the number of latent channels was increased to 32, paired with the same 8x downscaling factor, this VAE had a latent representation of $(32,64,64)$.

Due to using MSE, the model learned to produce blurry representations rather than sharp details, an example of which is shown here:
</div>

<img src=training_images/vae/512_reconstruction.webp height=800>

<div align='justify'>
As the diffusion model operates directly within the latent representation of the VAE, there is an upper limit on the detail which the generated images can have. The diffusion model will not be able to generate images which are outside of the capability of the VAE's decoder to produce. However, this VAE was kept as the reconstructions were still clearly flowers, which was the goal of this project.
</div>

## Latent Diffusion U-Net

<div align='justify'>

Latent Diffusion Model U-Nets were trained using both an epsilon prediction and v-prediction variant for each VAE. The U-Net used has an identical structure between tests, with 4 downscaling stages, followed by 4 upscaling stages. Spatial self-attention was added immediately before, and immediately after the bottleneck, where computational requirements would be lowest. As spatial self-attention scales quadratically with the number of tokens (for a 32x32 latent this would be $(32\times 32)^2 \approx 1\times10^6$ pairings), this placement was the least likely to cause computational issues
### 256x256 U-Net

An LDM U-Net was trained using both VAEs. Originally an epsilon prediction model was trained on the 256x256 VAE, however this implementation did not use self-attention, and did not share the architecture of the later U-Nets. Near the end, all models were retrained to use the same architecture. Below are a few samples of generations from the initial epsilon prediction model:
</div>

<img src=training_images/unet/eps_256x256.webp width=500>

Originally the model did not produce coherent outputs, as can be seen here: 
<img src=training_images/unet/eps_original.webp width=500>

<div align='justify'>

A large factor that contributed to these outputs was that the forward diffusion equation was implemented incorrectly at first, with the square roots not being properly added. This was fixed for all later models, this image being the only one shown that used the improper implementation.

As a limitation of epsilon prediction, when trying to recover the original latent during sampling, due to division by 0 (which is clamped to a small number to avoid this in the calculation of $\sigma$), the resulting value is undefined. This destabilizes the process, which results in images such as these:
</div>

<img src=training_images/unet/eps_256x256_noise.webp width=500>

<div align='justify'>
This could be artificially counteracted by capping the time-steps that the model is allowed to be sampled with. For example: These LDMs were trained with 1000 time-steps, however, by capping the maximum allowed sampling time-step to 900, this issue was visually solved:
</div>

<img src=training_images/unet/eps_capped_at_900.webp width=500>

<div align='justify'>

However, by doing this, there is a mismatch between the model's training, and what it is given during sampling. During sampling, a latent consisting of purely gaussian noise is given to the model. However, the model is being artificially  prevented from sampling at the maximum time-step (which is when it expects pure noise). 

To resolve this inconsistency, another LDM was trained using v-prediction, which can help mitigate this issue, allowing the model to sample using its entire learned time-step distribution. This happens because following the v-prediction formulation: $v = \sqrt{\bar{\alpha}} \epsilon  - \sqrt{1-\bar{\alpha}} x_0$, when $\bar{\alpha} = 0$ $v=-x_0$ which is still defined. The v-prediction model's outputs are below:<br> 
</div>
<img src=training_images/unet/v_pred_256x256.webp width=500> <br>
<img src=training_images/unet/v_pred_256x256_2.webp width=500>

<div align='justify'>
This model was trained using zero-terminal SNR, which rescales the noise schedule to ensure that at the final time-step, there is no information about the original image present. This makes a difference as without it, during training the final time-step may contain a small residual of the encoded image, while during inference the model is given pure gaussian noise at the final time-step.
</div>

### 512x512 U-Net

<div align='justify'>
Similarly, both an epsilon prediction and v-prediction U-Net were trained using the 512x512 VAE. As before, beginning with an epsilon prediction model. There was an issue in the original version of this model, where zero-terminal SNR was used during training. It was not set to false after training the v-prediction model, which resulted in images such as these:<br>
</div>
<img src=training_images/unet/eps_512x512_zero_terminal_snr.webp width=500>

<div align='justify'>
This is, in part, caused by the same issue as before, where to truly represent a zero-terminal SNR, the corresponding $\sigma = \infty$, which cannot be represented. 


A working epsilon prediction model at 512x512 was not completed. In order to determine if a change in the sampling code was the source of the noisy images, a v-prediction model was trained. This was with the hope that the different prediction target would resolve the issue (which would have suggested that the issue was with the epsilon prediction implementation, rather than the sampling).


Similarly to the epsilon prediction model, the v-prediction model did not perform very well. 
</div>
<img src=training_images/unet/v_pred_512x512.webp width=500>

<div align='justify'>
Some images are more visibly flowers than others, but they are all very noisy. It seems that the models trained in the larger latent space struggled much more to converge to meaningful generations. Perhaps, if allowed to train for longer, both models would have been able to generate coherent outputs.

A peculiarity was observed with both v-prediction models, they had a rather conservative prediction spread which remained mostly constant throughout training.
</div>

Results for each model:

512x512 v-prediction U-Net (Epoch 41)
||Target|Prediction|
|:---:|:---:|:---:|
| $\mu$|-0.000032 | -0.000049
|$\sigma$| 1.000015| 0.098156|

256x256 v-prediction U-Net (Epoch 41)
||Target|Prediction|
|:---:|:---:|:---:|
| $\mu$| 0.0002| 0.0001|
|$\sigma$|1.0001 | 0.1927|

<div align='justify'>

As shown, both v-prediction models has a significantly lower $\sigma$ prediction than the target, however the 256x256 model had a substantially larger $\sigma$ than the 512x512 model. Both of the v-prediction models were trained using min-SNR $\gamma$ weighting, with $\gamma = 5$ as in the respective paper [*Efficient Diffusion Training via Min-SNR Weighting Strategy* Hang et al.]. 

The 256x256 model was able to generate coherent outputs, even outperforming the corresponding epsilon prediction model. While the 512x512 v-prediction model was also able to outperform its 512x512 epsilon prediction counterpart, neither model's generations were satisfactory.

Similarly, the v-prediction models' MSE hardly decreased at all throughout training, even as their generations improved. This is in contrast to the epsilon prediction models, which were able to decrease their MSE quite quickly. The v-prediction implementation was able to reduce loss substantially in an overfitting test on a small sample of static latents. While this does not prove that the model learned anything rather than merely memorizing the latents, since image quality improved regardless, this wasn't investigated further and didn't affect this project's aim. This could potentially be investigated in the future, and it would be interesting to see if the v-prediction models would be able to reduce loss when trained on fresh latents given enough time.
</div>

## Current 
<div align='justify'>
All models were retrained to use their current architectures, this ensured that comparing their outputs was meaningful. The 512x512 models were kept as they were, as they were only trained using the most recent architecture, and, as their performance was unsatisfactory, they are only presented for completeness.
</div>

Final VAE Configurations:
| | Initial Hidden Dimensions | Latent Dimensions|
|:---:|:---:|:---:|
|256x256| 32 | 16|
|512x512| 64 | 32|

Final U-Net Configurations:
| Original Image Size |  Prediction Type | Initial Latent Input | Initial Hidden Dimensions | Time-Step Embedding Dimensions| 
|:---:|:---:|:---:|:---:|:---:|
|256x256| epsilon | 16 | 32 | 256|
|256x256| v | 16 | 32 | 256 |
|512x512 | epsilon| 32 | 32 | 256|
|512x512 | v | 32 | 32 | 256|

The initial latent input corresponds to the VAEs latent dimensions, which is why the 512x512 U-Nets used 32 rather than 16.

## Challenges and Further Work

<div align='justify'>
At some point after changing the U-Net architecture, epsilon prediction models no longer produced coherent outputs during sampling, while the v-prediction models' performance was not degraded too heavily. This was surprising as setting up the epsilon prediction models for training was a much easier task, which was more straightforward. It is possible that the changes made to ensure that v-prediction worked, may have interfered with the program's ability to train the epsilon prediction models.

Due to hardware limitations, as well as having a small dataset, it is likely that the 512x512 models were not trained for long enough, which may have contributed to their inability to match the level of the 256x256 models. In addition, the increase in the latent space representation of the VAE did not appear to help the models, which was reflected in their training logs, with the majority of VAE losses being concentrated in very few channels. As the reconstructions themselves appeared to be improving, this was not seen as an issue at the time, yet it may have had unforeseen consequences for the resulting U-Net which would be trained using the VAE's latent space. 

In further work, it would be beneficial to further explore this factor, specifically whether having more equal channel usage by the VAE would benefit the resulting U-Nets.
</div>

<img src=training_images/unet/sampled_latent_channel_usage.webp width=500><br>

<div align='justify'>
As seen above, it appears that the majority of information about the resulting image is concentrated in one channel, the effects of which weren't investigated.
</div>

## Image Gallery

Epsilon Prediction:
<img src=training_images/unet/eps_pred_grid1.webp>
<img src=training_images/unet/eps_pred_grid2.webp>

v-prediction:
<img src=training_images/unet/v_pred_grid1.webp>
<img src=training_images/unet/v_pred_grid2.webp>