from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import IFPipeline, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version, enable_gradient
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

from .my_unet_2d_condition import MyUNet2DConditionModel

@threestudio.register("deep-floyd-guidance")
class DeepFloydGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "DeepFloyd/IF-I-XL-v1.0"
        cmm_pretrained_model_name_or_path: str = ""
        cmm_schedule_pretrained_model_name_or_path: str = ""
        weighting_strategy: str = "dreamfusion"
        view_dependent_prompting: bool = True

        guidance_scale: float = 20.
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        grad_clip: Optional[Any] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])

        # FIXME: xformers error
        half_precision_weights: bool = True
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = True

        # a trick to rescale the camera to world postition
        c2w_scale: float = 1.0

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        self.min_step: Optional[int] = None
        self.max_step: Optional[int] = None
        self.cmm_min_step: Optional[int] = None
        self.cmm_max_step: Optional[int] = None
        self.grad_clip_val: Optional[float] = None
        threestudio.info(f"Loading Deep Floyd ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # Create model
        self.pipe = IFPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

        # force load the pretrained unet
        threestudio.info(f"Loading CCMUNet ...")
        self.cmm_unet = MyUNet2DConditionModel.from_pretrained(
            self.cfg.cmm_pretrained_model_name_or_path, 
            subfolder="unet",
            torch_dtype=torch.float16, 
            addition_embed_type='camera'
        ).to(self.device)
        self.cmm_unet.eval()
        enable_gradient(self.cmm_unet, enabled=False)
        threestudio.info(f"Loaded CCMUNet!")

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                threestudio.warn(
                    f"Use DeepFloyd with xformers may raise error, see https://github.com/deep-floyd/IF/issues/52 to track this problem."
                )
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        self.unet = self.pipe.unet.eval()

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )


        # for cmm sds
        self.cmm_scheduler = DDIMScheduler.from_pretrained(
            self.cfg.cmm_schedule_pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.cmm_num_train_timesteps = self.cmm_scheduler.config.num_train_timesteps
        self.cmm_alphas: Float[Tensor, "..."] = self.cmm_scheduler.alphas_cumprod.to(
            self.device
        )

        self.set_min_max_steps()  # set to default value

        threestudio.info(f"Loaded Deep Floyd!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.cmm_min_step = int(self.cmm_num_train_timesteps * min_step_percent)
        self.cmm_max_step = int(self.cmm_num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_cmm_unet(
        self,
        cmm_unet: MyUNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        camera2world: Optional[Float[Tensor, "B 12"]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return cmm_unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            camera2world=camera2world.to(self.weights_dtype),
        ).sample.to(input_dtype)
    
    def compute_cmm_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_condition: Float[Tensor, "B 4 3"],
    ):
        batch_size = elevation.shape[0]

        neg_guidance_weights = None
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.cmm_scheduler.add_noise(latents, noise, t)
            # prepare input
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            camera_condition_input = torch.cat([camera_condition.view(-1, 12)] * 2, dim=0)
            # pred noise
            noise_pred = self.forward_cmm_unet(
                self.cmm_unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                camera2world=camera_condition_input,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == "dreamfusion":
            # w(t), sigma_t^2
            w = (1 - self.cmm_alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.cmm_alphas[t] ** 0.5 * (1 - self.cmm_alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils
    
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        guidance_eval=False,
        is_cmm=False,
        rescale=True,
        **kwargs,
    ):
        if not is_cmm:
            batch_size = rgb.shape[0]

            rgb_BCHW = rgb.permute(0, 3, 1, 2)

            assert rgb_as_latents == False, f"No latent space in {self.__class__.__name__}"
            rgb_BCHW = rgb_BCHW * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )

            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

            if prompt_utils.use_perp_neg:
                (
                    text_embeddings,
                    neg_guidance_weights,
                ) = prompt_utils.get_text_embeddings_perp_neg(
                    elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
                )
                with torch.no_grad():
                    noise = torch.randn_like(latents)
                    latents_noisy = self.scheduler.add_noise(latents, noise, t)
                    latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                    noise_pred = self.forward_unet(
                        latent_model_input,
                        torch.cat([t] * 4),
                        encoder_hidden_states=text_embeddings,
                    )  # (4B, 6, 64, 64)

                noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
                noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(
                    3, dim=1
                )
                noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

                e_pos = noise_pred_text - noise_pred_uncond
                accum_grad = 0
                n_negative_prompts = neg_guidance_weights.shape[-1]
                for i in range(n_negative_prompts):
                    e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                    accum_grad += neg_guidance_weights[:, i].view(
                        -1, 1, 1, 1
                    ) * perpendicular_component(e_i_neg, e_pos)

                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    e_pos + accum_grad
                )
            else:
                neg_guidance_weights = None
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
                )
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # add noise
                    noise = torch.randn_like(latents)  # TODO: use torch generator
                    latents_noisy = self.scheduler.add_noise(latents, noise, t)
                    # pred noise
                    latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                    noise_pred = self.forward_unet(
                        latent_model_input,
                        torch.cat([t] * 2),
                        encoder_hidden_states=text_embeddings,
                    )  # (2B, 6, 64, 64)

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
                noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            """
            # thresholding, experimental
            if self.cfg.thresholding:
                assert batch_size == 1
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
                noise_pred = custom_ddpm_step(self.scheduler,
                    noise_pred, int(t.item()), latents_noisy, **self.pipe.prepare_extra_step_kwargs(None, 0.0)
                )
            """

            if self.cfg.weighting_strategy == "dreamfusion":
                # w(t), sigma_t^2
                w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "uniform":
                w = 1
            elif self.cfg.weighting_strategy == "fantasia3d":
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
                )

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            # loss = SpecifyGradient.apply(latents, grad)
            # SpecifyGradient is not straghtforward, use a reparameterization trick instead
            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

            guidance_out = {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }

            if guidance_eval:
                guidance_eval_utils = {
                    "use_perp_neg": prompt_utils.use_perp_neg,
                    "neg_guidance_weights": neg_guidance_weights,
                    "text_embeddings": text_embeddings,
                    "t_orig": t,
                    "latents_noisy": latents_noisy,
                    "noise_pred": torch.cat([noise_pred, predicted_variance], dim=1),
                }
                guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
                texts = []
                for n, e, a, c in zip(
                    guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
                ):
                    texts.append(
                        f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                    )
                guidance_eval_out.update({"texts": texts})
                guidance_out.update({"eval": guidance_eval_out})

            return guidance_out
        else:
            batch_size = rgb.shape[0]

            rgb_BCHW = rgb.permute(0, 3, 1, 2)
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = F.interpolate(
                    rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                )
            else:
                rgb_BCHW_512 = F.interpolate(
                    rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
                )
                # encode image into latents with vae
                latents = self.encode_images(rgb_BCHW_512)
            

            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

            camera_condition = c2w

            # our camera_condition type, camera to world 
            camera_condition[:, :3, :4] = torch.tensor([[0, 1, 0],
                                                        [-1, 0, 0],
                                                        [0, 0, 1]], dtype=torch.float).to(camera_condition.device) @ camera_condition[:, :3, :4]
            camera_condition[:, :3, 3] *= self.cfg.c2w_scale

            camera_condition = camera_condition[:, :3, :4] # we only use 3 x 4 matrix for camera

            grad, _ = self.compute_cmm_grad_sds(
                latents, t, prompt_utils, elevation, azimuth, camera_distances, camera_condition
            )
            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            # loss = SpecifyGradient.apply(latents, grad)
            # SpecifyGradient is not straghtforward, use a reparameterization trick instead
            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

            guidance_out = {
                'loss_cmm_sds': loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }

            return guidance_out
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]
        if use_perp_neg:
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 6, 64, 64)

            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(
                3, dim=1
            )
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (2B, 6, 64, 64)

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return torch.cat([noise_pred, predicted_variance], dim=1)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = (latents_noisy[:bs] / 2 + 0.5).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1]
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = (latents_1step / 2 + 0.5).permute(0, 2, 3, 1)
        imgs_1orig = (pred_1orig / 2 + 0.5).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = (latents_final / 2 + 0.5).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

