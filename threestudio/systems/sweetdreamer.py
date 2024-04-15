import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("sweetdreamer-system")
class SweeetDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        cmm_prompt_processor_type: str = ""
        cmm_prompt_processor: dict = field(default_factory=dict)

        before_start_app_weight: float = 1.0
        start_app: int = 0
        app_weight: float = 1.0
        end_app: int = 1000
        after_end_app_weight: float = 1.0

        before_start_cmm_weight: float = 0.0
        start_cmm: int = 0
        cmm_weight: float = 0.0
        end_cmm: int = 1000
        after_end_cmm_weight: float = 0.0

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.cmm_prompt_processor = threestudio.find(self.cfg.cmm_prompt_processor_type)(
            self.cfg.cmm_prompt_processor
        )
        self.cmm_prompt_utils = self.cmm_prompt_processor()

    def forward(self, batch: Dict[str, Any], true_global_step=0) -> Dict[str, Any]:
        render_out = self.renderer(**batch, true_global_step=true_global_step)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        out = self(batch, self.true_global_step)


        if self.true_global_step >= self.cfg.start_app:
            guidance_inp = out["comp_rgb"]
            if self.cfg.guidance_type == "stable-diffusion-controlnet-guidance":
                guidance_out = self.guidance(
                    guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False, cond_rgb=out["comp_normal_viewspace"]
                )
            else:
                guidance_out = self.guidance(
                    guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False,
                )

        if self.true_global_step >= self.cfg.start_cmm:
            cmm_guidance_inp = out["comp_cmm"]
            cmm_guidance_out = self.guidance(
                cmm_guidance_inp, self.cmm_prompt_utils, **batch, rgb_as_latents=True, is_cmm=True
            )

        loss = 0.0

        if self.true_global_step >= self.cfg.start_app:
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    if self.true_global_step < self.cfg.start_app:
                        weight = self.cfg.before_start_app_weight
                    elif self.true_global_step < self.cfg.end_app:
                        weight = self.cfg.app_weight
                    else:
                        weight = self.cfg.after_end_app_weight
                    self.log(f"train/raw_weight", weight)
                    loss += value * weight

        if self.true_global_step >= self.cfg.start_cmm:
            for name, value in cmm_guidance_out.items():
                self.log(f"train/cmm_{name}", value)
                if name.startswith("loss_"):
                    if self.true_global_step < self.cfg.start_cmm:
                        weight = self.cfg.before_star_cmm_weight
                    elif self.true_global_step < self.cfg.end_cmm:
                        weight = self.cfg.cmm_weight
                    else:
                        weight = self.cfg.after_end_cmm_weight
                    self.log(f"train/cmm_weight", weight)
                    loss += value * weight

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if "z_variance" in out:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        # sdf loss
        if "sdf_grad" in out:
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
            self.log("train/inv_std", out["inv_std"], prog_bar=True)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_cmm"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_cmm" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )


    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_cmm"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_cmm" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
