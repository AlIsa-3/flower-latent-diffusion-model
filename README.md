# Flower Image Generation Via Latent Diffusion Model

<div align='justify'>
Latent diffusion models (LDMs) were implemented in PyTorch and trained on the Oxford Flowers 102 dataset. Multiple variants were trained, each with the goal of improving a shortcoming of the previous model. Below is an overview of the process, including the generated outputs of the individual models. 
</div>

## Variational Autoencoder (VAE)

<div align='justify'>

Initially a relatively shallow VAE was trained at $256 \times 256$ to experiment with the architecture. The architecture was iteratively modified and evaluated, with the aim of investigating whether increasing the model capacity would improve the generation quality, while still remaining relatively inexpensive to use. All hidden dimension values within the model are derived as multiples of the initial hidden dimension count passed when constructing the VAE class. 
</div>

Initial VAE Configuration:
| | Initial Hidden Dimensions | Latent Dimensions|
|:---:|:---:|:---:|
|$256 \times 256$ | 16 | 8|
|$512 \times 512$ | 64 | 32|

<div align='justify'>

Both VAE models are convolutional VAEs containing residual blocks and using SiLU activations. They both have a downscaling factor of 8 (so a $256 \times 256$  image gets compressed to a $32 \times 32$  latent representation), however their channel counts and latent representation are different. 

Adding spatial self-attention was something which was experimented with in the VAE, both within the encoder and the decoder near the bottleneck; however, the improvement in reconstruction quality was not significant enough to continue using it. In addition, due to hardware limitations, the computational cost of adding attention meant that the spatial compression factor had to be increased to 16, causing the model to lose more spatial representation. 

The $512 \times 512$  VAE was trained to determine if having a larger image would allow the model to create more detailed flower images. If the diffusion models had been trained in pixel-space, this increase may have been too much for the hardware to handle, however, by training in latent-space this increase was only from $32 \times 32$  to $64 \times 64$, which could still be computed in a reasonable amount of time.
</div>

### $256 \times 256$  VAE

<div align='justify'>

For the VAE trained at $256 \times 256$  px resolution, the model struggled to reconstruct the finer details of the flower images (such as the flower centers), however it was able to reconstruct the large-scale structures quite well. It was originally trained with mean squared error (MSE), however that tended to produce overly blurry reconstructions. To try and combat this, the model was retrained using mean absolute error, with the goal of preserving the finer visual details.

Below is a sample of its performance across different epochs.
</div>
<img src=training_images/vae/32x32_32hidden_8latent.webp width=1000>

<div align='justify'>

The top row is the input image, and the bottom row is the reconstruction. Each column represents every 10th epoch (1, 11, 21, etc).

One potential cause of the difficulty in reconstruction was the size of the model. This VAE's latent representation for the $256 \times 256$  image had a dimensionality of $(8,32,32)$, which meant that it may not have had enough capacity to represent the information needed.
</div>

### $512 \times 512$  VAE
<div align='justify'>

For the $512 \times 512$  VAE, the model was trained using MSE, unlike in the $256 \times 256$  VAE, however it had the same model architecture. In order to try and mitigate the issues of not having enough model capacity, the number of latent channels was increased to 32. When paired with the same 8x downscaling factor, this gave the VAE a latent representation of $(32,64,64)$.

Due to using MSE, the model learned to produce blurry representations rather than sharp details, an example of which is shown here:
</div>

<img src=training_images/vae/512_reconstruction.webp height=800>

<div align='justify'>

As the diffusion model operates directly within the latent representation of the VAE, there is an upper limit on the detail which the generated images can have. The diffusion model will not be able to generate images which are outside of the capability of the VAE's decoder to produce. However, this VAE was kept as the reconstructions were still clearly flowers, which was the goal of this project.
</div>

## Latent Diffusion U-Net

<div align='justify'>

Latent Diffusion Model U-Nets were trained using both an epsilon-prediction and v-prediction variant for each VAE. The U-Net architecture was kept identical between tests, with 4 downscaling stages, followed by 4 upscaling stages. Spatial self-attention was added immediately before, and immediately after the bottleneck, where computational requirements would be lowest due to having the smallest spatial dimensions. As spatial self-attention scales quadratically with the number of tokens (for a $32 \times 32$  latent this would be $(32\times 32)^2 \approx 1\times10^6$ pairings), this placement was the least likely to cause computational issues.
### $256 \times 256$  U-Net

Both epsilon-prediction and v-prediction U-Nets were trained using both VAEs. Originally an epsilon-prediction model was trained on the $256 \times 256$  VAE, however this implementation did not use self-attention and did not share the architecture of the later U-Nets. Later in the project, all models were retrained to use the same architecture. Below are a few samples generated by the initial epsilon-prediction model:
</div>

<img src=training_images/unet/eps_256x256.webp width=500>

Originally the model did not produce coherent outputs, as can be seen here: 
<img src=training_images/unet/eps_original.webp width=500>

<div align='justify'>

A large factor that contributed to these outputs was an incorrect implementation of the forward diffusion equation, with the square roots being omitted. This was fixed for all later models, this image being the only one shown that used the improper implementation.

During sampling of an epsilon-prediction model, the sigma parameterization is used to recover the denoised latent. As the SNR approaches zero, the corresponding sigma value approaches infinity. To avoid this, the $\bar{\alpha}$ values are clamped to a small positive value when calculating sigma. However, this still produces a very large sigma, which amplifies errors in the predicted noise and can destabilize the sampling process. 
</div>

<img src=training_images/unet/eps_256x256_noise.webp width=500>

<div align='justify'>
One possible workaround is by capping the time steps that the model is allowed to be sampled with. For example, these LDMs were trained with 1000 time steps, however, by capping the maximum allowed sampling time step to 900, this issue was visually solved:
</div>

<img src=training_images/unet/eps_capped_at_900.webp width=500>

<div align='justify'>

However, by doing this, there is a mismatch between the model's training and what it is given during sampling. During sampling, a latent consisting of purely gaussian noise is given to the model. However, the model is being artificially prevented from sampling at the maximum time step (which is when it expects pure noise). 

To resolve this inconsistency, another LDM was trained using v-prediction, which can help mitigate this issue, allowing the model to sample using its entire learned time step distribution. This happens because following the v-prediction formulation: $v = \sqrt{\bar{\alpha}} \epsilon  - \sqrt{1-\bar{\alpha}} x_0$, when $\bar{\alpha} = 0$ $v=-x_0$ which is still defined. The v-prediction model's outputs are below:<br> 
</div>
<img src=training_images/unet/v_pred_256x256.webp width=500> <br>
<img src=training_images/unet/v_pred_256x256_2.webp width=500>

<div align='justify'>
This model was trained using zero-terminal SNR, which rescales the noise schedule to ensure that at the final time step, there is no information about the original image present. This makes a difference as without it, during training the final time step may contain a small residual of the encoded image, while during inference the model is given pure gaussian noise at the final time step.
</div>

### $512 \times 512$  U-Net

<div align='justify'>

Similarly, both an epsilon-prediction U-Net and a v-prediction U-Net were trained using the $512 \times 512$  VAE. As before, beginning with an epsilon-prediction model. The original version of this model has an issue where zero-terminal SNR was unintentionally used during training. It was not set to false after training the v-prediction model, which resulted in images such as these:<br>
</div>
<img src=training_images/unet/eps_512x512_zero_terminal_snr.webp width=500>

<div align='justify'>

This may be caused, in part, by the same issue discussed earlier: in the sigma parameterization used for sampling, the corresponding sigma value approaches infinity as the SNR approaches zero. 


A working epsilon-prediction model at $512 \times 512$  was not completed. In order to determine if a change in the sampling code was the source of the noisy images, a v-prediction model was trained. This was with the hope that the different prediction target would resolve the issue (which would have suggested that the issue lay in the epsilon-prediction implementation, rather than the sampling).


Similarly to the epsilon-prediction model, the v-prediction model did not perform very well. 
</div>
<img src=training_images/unet/v_pred_512x512.webp width=500>

<div align='justify'>
Some images are more visibly flowers than others, but they are all very noisy. It seems that the models trained in the larger latent space struggled much more to converge to meaningful generations. Perhaps, if allowed to train for longer, both models would have been able to generate coherent outputs.

A peculiarity was observed with both v-prediction models: they had a rather conservative prediction spread which remained mostly constant throughout training. These values were all calculated across the validation set.
</div>

Results for each model:

$512 \times 512$  v-prediction U-Net (Epoch 41)
||Target|Prediction|
|:---:|:---:|:---:|
| $\mu$|-0.000032 | -0.000049
|$\sigma$| 1.000015| 0.098156|

$256 \times 256$  v-prediction U-Net (Epoch 41)
||Target|Prediction|
|:---:|:---:|:---:|
| $\mu$| 0.0002| 0.0001|
|$\sigma$|1.0001 | 0.1927|

<div align='justify'>

As shown, both v-prediction models have a significantly lower $\sigma$ prediction than the target, however the $256 \times 256$  model had a substantially larger $\sigma$ than the $512 \times 512$  model. Both of the v-prediction models were trained using min-SNR $\gamma$ weighting, with $\gamma = 5$ as in the respective paper [*Efficient Diffusion Training via Min-SNR Weighting Strategy* Hang et al.]. 

The $256 \times 256$  model was able to generate coherent outputs, even outperforming the corresponding epsilon-prediction model. While the $512 \times 512$  v-prediction model was also able to outperform its $512 \times 512$  epsilon-prediction counterpart, neither model's generations were satisfactory.

Similarly, the v-prediction models' MSE hardly decreased at all throughout training, even as their generations improved. This is in contrast to the epsilon-prediction models, which were able to decrease their MSE quite quickly. The v-prediction implementation was able to reduce loss substantially in an overfitting test on a small sample of static latents. While this does not prove that the model learned anything rather than merely memorizing the latents, since image quality improved regardless, this wasn't investigated further and didn't affect this project's aim. This could be investigated in future work, and it would be interesting to see if the v-prediction models would be able to reduce loss when trained on fresh latents given enough time.
</div>

## Current 
<div align='justify'>

All models were retrained to use their current architectures, this ensured that comparing their outputs was meaningful. The $512 \times 512$ models were left unchanged, as they were only trained using the most recent architecture, and, as their performance was unsatisfactory, they are only presented for completeness.
</div>
<br>
Final VAE Configurations:

| | Initial Hidden Dimensions | Latent Dimensions|
|:---:|:---:|:---:|
|$256 \times 256$ | 32 | 16|
|$512 \times 512$ | 64 | 32|

Final U-Net Configurations:
| Original Image Size |  Prediction Type | Latent Channels | Initial Hidden Dimensions | Time Step Embedding Dimensions| 
|:---:|:---:|:---:|:---:|:---:|
|$256 \times 256$ | epsilon | 16 | 32 | 256|
|$256 \times 256$ | v | 16 | 32 | 256 |
|$512 \times 512$  | epsilon| 32 | 32 | 256|
|$512 \times 512$  | v | 32 | 32 | 256|

The initial latent input corresponds to the VAEs' latent dimensions, which is why the $512 \times 512$  U-Nets used 32 rather than 16.

### Computed Metrics
<div align='justify'>

All three models were tested using around 800 real and 800 generated images. Both U-Nets used the same random seeds for generating images, while the VAE reconstructions only used the latent mean ($\mu$) to avoid introducing stochasticity from reparameterization.

The Fréchet Inception Distance (FID) was calculated. The FID exhibits finite-sample bias for small datasets, which may affect how models appear to perform, so the Kernel Inception Distance (KID) was also calculated as it is less susceptible to these biases.

The VAE and U-Net test sets are not the same, due to having different seeds used to split the dataset. However, both U-Nets use the same testing set. Therefore, the results below indicate different things. The VAE scores indicate how closely the feature distributions of the reconstructed images match those of the real ones. The U-Net scores measure the same thing between the generated images and the real images.
</div>

FID:
|Model|FID|
|:---:|:---:|
|VAE|$82.7439$|
|v-prediction U-Net| $140.3884$|
|epsilon-prediction U-Net| $202.5600$|

KID:
|Model| KID Mean | KID standard deviation|
|:---:|:---:|:---:|
|VAE| $0.04277$ |$0.00632$ |
|v-prediction U-Net|$0.1307$ |$0.0115$|
|epsilon-prediction U-Net| $0.2201$|$0.0134$|

<div align='justify'>

For the FID score, lower values are better, with the best possible score being $0$. The v-prediction model scoring $\approx 140$ while the epsilon-prediction model scores $\approx 202$ clearly indicates that the v-prediction model's generations are closer in distribution to true images of flowers (v-prediction FID is $\approx 30.7\\%$ lower than epsilon-prediction).  

For the KID score, the ideal population KID Mean is also $0$. The v-prediction model clearly performs better than the epsilon-prediction model, with $0.1307\pm 0.0115$ compared to $0.2201 \pm 0.0134$, this is an $\approx 40.6\\%$ reduction from the epsilon-prediction model.

Both metrics indicate that the v-prediction model performs better than the epsilon-prediction model in this experiment. This agreement provides additional support for this ranking, although both rely on Inception features.
</div>

## Challenges and Further Work

<div align='justify'>

While the goal of the project was to generate images that could reasonably be classified as flowers, the FID and KID values measure how closely the distribution of generated images matches that of real flower images. These values are influenced by visual characteristics such as color, the sharpness of the images, and overall image structure. 

Due to hardware limitations, further improvements in generation quality could not be reasonably achieved within the scope of this project. However, with access to better hardware and additional training time, there may be an opportunity to improve the generation quality further. This is something that could be explored in future work.

At some point after changing the U-Net architecture, epsilon-prediction models no longer produced coherent outputs during sampling, while the v-prediction models' performance was not degraded to the same extent. This was surprising, as setting up the epsilon-prediction models for training was more straightforward than setting up v-prediction, offering fewer opportunities for implementation errors. It is possible that the changes made to ensure that v-prediction worked may have interfered with the training or sampling of the epsilon-prediction models.

Due to the aforementioned hardware limitations, as well as having a small dataset, it is likely that the $512 \times 512$  models were not trained for long enough, which may have contributed to their inability to match the performance of the $256 \times 256$  models. In addition, the increase in the latent dimensionality of the VAE did not appear to help the models. This was reflected in their training logs, with the majority of the KL divergence being concentrated in very few channels. This concentration may have contributed to the lower performance. As the reconstructions themselves appeared to be improving, this was not seen as an issue at the time, yet it may have had unforeseen consequences for the resulting U-Nets, which were trained using the VAE's latent space. 

In further work, it would be beneficial to further explore this factor, specifically whether encouraging more balanced channel usage by the VAE would benefit the resulting U-Nets. In addition, testing additional latent dimensionalities could help determine how they affect generation quality. 
</div>

<img src=training_images/unet/sampled_latent_channel_usage.webp width=500><br>

<div align='justify'>
As seen above, the sampled latent representation appears to make uneven use of its channels. The majority of the visible spatial structure in the latent representation appears to be concentrated in one channel, while the others appear to contain mostly noise. This may indicate that the VAE has learned to concentrate a disproportionate amount of its spatial information into a small number of channels
</div>

## Image Gallery

Epsilon-prediction:
<img src=training_images/unet/eps_pred_grid1.webp>
<img src=training_images/unet/eps_pred_grid2.webp>

v-prediction:
<img src=training_images/unet/v_pred_grid1.webp>
<img src=training_images/unet/v_pred_grid2.webp>