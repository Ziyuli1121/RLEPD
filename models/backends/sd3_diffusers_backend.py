"""
SD3 backend adapter that wraps the official diffusers pipeline.

The adapter keeps all diffusers-specific logic (prompt encoding, CFG,
VAE decode, etc.) inside a single module so that the rest of the RLEPD
codebase can keep using the classic `(net(x, t, condition=..., ...))`
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    from diffusers import StableDiffusion3Pipeline
except ImportError as exc:  # pragma: no cover - handled explicitly for clarity
    raise ImportError(
        "StableDiffusion3Pipeline is unavailable. Install diffusers>=0.29.0 "
        "and make sure the SD3 gate was approved."
    ) from exc


PromptType = Union[str, Sequence[str]]


@dataclass
class SD3Conditioning:
    """Container for cached text embeddings and CFG metadata."""

    prompt_embeds: torch.FloatTensor
    pooled_prompt_embeds: torch.FloatTensor
    negative_prompt_embeds: Optional[torch.FloatTensor]
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor]
    guidance_scale: float
    num_images_per_prompt: int = 1

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1.0 and self.negative_prompt_embeds is not None


class SD3DiffusersBackend(nn.Module):
    """
    Thin adapter over StableDiffusion3Pipeline.

    Parameters mirror the official pipeline so that any later options
    (LoRA, IP-adapter, etc.) can be forwarded without touching RLEPD
    solver code.
    """

    def __init__(
        self,
        model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        *,
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        guidance_scale: float = 7.5,
        enable_model_cpu_offload: bool = False,
        max_sequence_length: int = 256,
        clip_skip: Optional[int] = None,
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        use_safetensors: Optional[bool] = True,
        token: Optional[str] = None,
        pipeline_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.default_guidance_scale = float(guidance_scale)
        self.max_sequence_length = max_sequence_length
        self.clip_skip = clip_skip
        self.device = torch.device(device) if isinstance(device, str) else device

        load_kwargs = {
            "torch_dtype": torch_dtype,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        if variant is not None:
            load_kwargs["variant"] = variant
        if use_safetensors is not None:
            load_kwargs["use_safetensors"] = use_safetensors
        if token is not None:
            load_kwargs["token"] = token
        if pipeline_kwargs:
            load_kwargs.update(pipeline_kwargs)

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        if enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

        # Public attributes consumed by sampler/sample.py.
        transformer_cfg = self.pipeline.transformer.config
        self.img_resolution = transformer_cfg.sample_size
        self.img_channels = transformer_cfg.in_channels
        self.label_dim = False
        self.backend = "sd3"
        self.backend_config = {}

        scheduler = self.pipeline.scheduler
        self.sigma_min = float(getattr(scheduler, "sigma_min", 0.0))
        self.sigma_max = float(getattr(scheduler, "sigma_max", 1.0))
        self.flow_shift = float(getattr(scheduler, "shift", 1.0))

    # --------------------------------------------------------------------- #
    # Conditioning helpers

    def prepare_condition(
        self,
        prompt: PromptType,
        negative_prompt: Optional[PromptType] = "",
        *,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: Optional[int] = None,
        clip_skip: Optional[int] = None,
    ) -> SD3Conditioning:
        """
        Cache text embeddings for later reuse.

        Returns:
            SD3Conditioning with positive/negative embeddings and CFG scale.
        """
        gs = float(guidance_scale) if guidance_scale is not None else self.default_guidance_scale
        max_seq = max_sequence_length or self.max_sequence_length
        clip_skip = clip_skip if clip_skip is not None else self.clip_skip

        do_cfg = gs > 1.0
        prompt_seq = prompt
        negative_seq = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            self.pipeline.encode_prompt(
                prompt=prompt_seq,
                prompt_2=None,
                prompt_3=None,
                device=self.pipeline._execution_device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_seq,
                negative_prompt_2=None,
                negative_prompt_3=None,
                clip_skip=clip_skip,
                max_sequence_length=max_seq,
            )
        )

        return SD3Conditioning(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds if do_cfg else None,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_cfg else None,
            guidance_scale=gs,
            num_images_per_prompt=num_images_per_prompt,
        )

    # --------------------------------------------------------------------- #
    # Main call used by solvers

    def forward(  # type: ignore[override]
        self,
        latents: torch.FloatTensor,
        t: Union[float, torch.Tensor],
        *,
        condition: Optional[SD3Conditioning] = None,
        unconditional_condition: Optional[SD3Conditioning] = None,
    ) -> torch.FloatTensor:
        """
        Compute `x - t * v_theta` so solvers can keep using `(x - denoised)/t`.
        """
        if condition is None:
            raise ValueError("SD3DiffusersBackend requires a conditioning object.")
        if unconditional_condition is not None:
            raise ValueError("SD3 backend does not use `unconditional_condition`; pass guidance via prepare_condition.")

        exec_device = self.pipeline._execution_device
        latent_dtype = self.pipeline.transformer.dtype
        latents = latents.to(device=exec_device, dtype=latent_dtype)
        batch_size = latents.shape[0]

        timestep = self._format_timesteps(t, batch_size, latents.dtype, latents.device)

        do_cfg = condition.do_classifier_free_guidance
        prompt_embeds = condition.prompt_embeds
        pooled_prompt_embeds = condition.pooled_prompt_embeds

        if do_cfg:
            if condition.negative_prompt_embeds is None or condition.negative_pooled_prompt_embeds is None:
                raise ValueError("Negative prompt embeddings are required when guidance_scale > 1.0.")
            prompt_embeds = torch.cat([condition.negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [condition.negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
        timestep = (
            torch.cat([timestep, timestep], dim=0)
            if do_cfg
            else timestep
        )

        noise_pred = self.pipeline.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            velocity = noise_pred_uncond + condition.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            velocity = noise_pred

        # Convert velocity into the "denoised" quantity expected by solvers.
        t_reshaped = self._reshape_timestep(t, latents)
        denoised = latents - t_reshaped * velocity
        return denoised

    __call__ = forward  # keep legacy invocation style

    # ------------------------------------------------------------------ #
    # Utilities

    def vae_decode(self, latents: torch.FloatTensor, output_type: str = "tensor") -> Union[torch.Tensor, Sequence]:
        """
        Decode latent samples into image space.
        """
        latents = latents.to(self.pipeline._execution_device, dtype=self.pipeline.transformer.dtype)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        if output_type == "tensor":
            return image
        return self.pipeline.image_processor.postprocess(image, output_type=output_type)

    @staticmethod
    def _format_timesteps(
        t: Union[float, torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([float(t)], device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        elif t.ndim == 1 and t.shape[0] != batch_size:
            raise ValueError(f"Timestep tensor shape {t.shape} does not match batch size {batch_size}.")
        elif t.ndim > 1:
            t = t.reshape(batch_size)
        return t

    @staticmethod
    def _reshape_timestep(t: Union[float, torch.Tensor], latents: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t_tensor = torch.tensor(float(t), device=latents.device, dtype=latents.dtype)
        else:
            t_tensor = t.to(device=latents.device, dtype=latents.dtype)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.view(1).repeat(latents.shape[0])
        t_tensor = t_tensor.view(latents.shape[0], 1, 1, 1)
        return t_tensor
