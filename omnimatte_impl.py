import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union, Dict, Any
from contextlib import contextmanager

from diffusers import LTXConditionPipeline, AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    retrieve_latents, LTXVideoCondition, linear_quadratic_schedule, retrieve_timesteps
)
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks


# ==============================================================================
# MPS SAFETY UTILITIES
# ==============================================================================

def _is_mps(device) -> bool:
    """Check if a device is Apple Silicon MPS."""
    if isinstance(device, torch.device):
        return device.type == "mps"
    return str(device) == "mps"


def _safe_randn_tensor(shape, generator=None, device=None, dtype=None):
    """
    MPS-safe wrapper around randn_tensor.

    On MPS, torch.Generator can be unreliable in certain PyTorch versions.
    If the generator is on CPU but target device is MPS, we generate on CPU
    and move — which is what diffusers does internally, but some older
    diffusers versions mishandle this path.  This wrapper ensures it.
    """
    if generator is not None and _is_mps(device):
        # Always generate on CPU when targeting MPS, then move
        tensor = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
        return tensor.to(device)
    return randn_tensor(shape, generator=generator, device=device, dtype=dtype)


def _empty_cache_if_mps(device):
    """Flush MPS command buffer and free cached allocations."""
    if _is_mps(device):
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.synchronize()
                torch.mps.empty_cache()
        except Exception:
            pass


# ==============================================================================
# 1. OMNIMATTE PIPELINE (extends LTXConditionPipeline — the correct base class)
# ==============================================================================

class OmnimatteZeroPipeline(LTXConditionPipeline):
    """
    OmnimatteZero pipeline extending LTXConditionPipeline with custom
    my_prepare_latents and my_call for mask-guided video inpainting.

    Based on: https://github.com/dvirsamuel/OmnimatteZero/blob/main/OmnimatteZero.py
    """

    def my_prepare_latents(
            self,
            conditions: Optional[List[torch.Tensor]] = None,
            condition_strength: Optional[List[float]] = None,
            condition_frame_index: Optional[List[int]] = None,
            batch_size: int = 1,
            num_channels_latents: int = 128,
            height: int = 512,
            width: int = 704,
            num_frames: int = 161,
            num_prefix_latent_frames: int = 2,
            sigma: Optional[torch.Tensor] = None,
            latents: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        # MPS: use safe randn to avoid generator device mismatch
        noise = _safe_randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if latents is not None and sigma is not None:
            if latents.shape != shape:
                raise ValueError(
                    f"Latents shape {latents.shape} does not match expected shape {shape}."
                )
            latents = latents.to(device=device, dtype=dtype)
            sigma = sigma.to(device=device, dtype=dtype)
            latents = sigma * noise + (1 - sigma) * latents
        else:
            latents = noise.clone()

        conditioning_mask_shape = None

        if len(conditions) > 0:
            condition_latent_frames_mask = torch.zeros(
                (batch_size, num_latent_frames), device=device, dtype=torch.float32
            )

            extra_conditioning_latents = []
            extra_conditioning_video_ids = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0

            # conditions[0] = video tensor, conditions[1] = mask tensor
            data, mask, strength, frame_index = (
                conditions[0], conditions[1], condition_strength[0], condition_frame_index[0]
            )

            # Encode video to latent space
            condition_latents = retrieve_latents(self.vae.encode(data), generator=generator)
            print(f"[OmnimatteZero] condition_latents shape: {condition_latents.shape}")
            condition_latents = self._normalize_latents(
                condition_latents, self.vae.latents_mean, self.vae.latents_std
            ).to(device, dtype=dtype)

            # Encode mask: amplify values before VAE encoding to get clear latent footprint
            mask[mask < 0] = 0
            mask[mask > 0] = 100
            mask_latents = retrieve_latents(self.vae.encode(mask), generator=generator)

            # Determine mask shape in latent space
            conditioning_mask_shape = torch.logical_or(
                mask_latents[0].mean(0) > 0.01,
                mask_latents[0].mean(0) < -0.01
            )
            conditioning_mask_shape = conditioning_mask_shape.type(dtype).unsqueeze(0).unsqueeze(0)

            num_cond_frames = condition_latents.size(2)

            # Blend condition latents into noise for unmasked regions
            latents[:, :, :num_cond_frames] = torch.lerp(
                latents[:, :, :num_cond_frames], condition_latents, strength
            )
            condition_latent_frames_mask[:, :num_cond_frames] = strength

        # Prepare video positional IDs
        video_ids = self._prepare_video_ids(
            batch_size, num_latent_frames, latent_height, latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
            device=device,
        )

        if len(conditions) > 0:
            conditioning_mask = (1 - conditioning_mask_shape[0].reshape(1, -1))
        else:
            conditioning_mask, extra_conditioning_num_latents = None, 0

        video_ids = self._scale_video_ids(
            video_ids,
            scale_factor=self.vae_spatial_compression_ratio,
            scale_factor_t=self.vae_temporal_compression_ratio,
            frame_index=0,
            device=device,
        )

        # KEY: masked regions get noise, unmasked keep condition latents
        if conditioning_mask_shape is not None:
            latents = latents * (1 - conditioning_mask_shape) + noise * conditioning_mask_shape

        latents = self._pack_latents(
            latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )

        if len(conditions) > 0 and len(extra_conditioning_latents) > 0:
            latents = torch.cat([*extra_conditioning_latents, latents], dim=1)
            video_ids = torch.cat([*extra_conditioning_video_ids, video_ids], dim=2)
            conditioning_mask = torch.cat([*extra_conditioning_mask, conditioning_mask], dim=1)

        return latents, conditioning_mask, video_ids, extra_conditioning_num_latents

    @torch.no_grad()
    def my_call(
            self,
            conditions=None,
            prompt=None,
            negative_prompt=None,
            height: int = 512,
            width: int = 704,
            num_frames: int = 161,
            frame_rate: int = 25,
            num_inference_steps: int = 50,
            guidance_scale: float = 3.0,
            image_cond_noise_scale: float = 0.15,
            generator=None,
            latents=None,
            denoise_strength: float = 1.0,
            decode_timestep: float = 0.0,
            decode_noise_scale=None,
            output_type: str = "pil",
            max_sequence_length: int = 256,
            **kwargs,
    ):
        self._guidance_scale = guidance_scale
        self._interrupt = False
        self._current_timestep = None

        batch_size = 1
        device = self._execution_device

        # 1. Encode prompt
        (
            prompt_embeds, prompt_attention_mask,
            negative_prompt_embeds, negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt or "",
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 2. Prepare conditioning tensors from LTXVideoCondition pairs
        conditioning_tensors = []
        is_conditioning = conditions is not None and len(conditions) > 0

        if is_conditioning:
            # Check if conditions are LTXVideoCondition or raw tensors
            if isinstance(conditions[0], LTXVideoCondition):
                vae_dtype = self.vae.dtype
                strength = [c.strength for c in conditions]
                frame_index = [c.frame_index for c in conditions]

                for cond in conditions:
                    if cond.video is not None:
                        condition_tensor = self.video_processor.preprocess_video(cond.video, height, width)
                        num_frames_input = condition_tensor.size(2)
                        num_frames_output = self.trim_conditioning_sequence(
                            cond.frame_index, num_frames_input, num_frames
                        )
                        condition_tensor = condition_tensor[:, :, :num_frames_output]
                        condition_tensor = condition_tensor.to(device, dtype=vae_dtype)
                    elif cond.image is not None:
                        condition_tensor = (
                            self.video_processor.preprocess(cond.image, height, width)
                            .unsqueeze(2).to(device, dtype=vae_dtype)
                        )
                    else:
                        raise ValueError("LTXVideoCondition must have video or image")

                    if condition_tensor.size(2) % self.vae_temporal_compression_ratio != 1:
                        # Fix frame count to be VAE-compatible
                        valid_f = fix_num_frames_for_vae(
                            condition_tensor.size(2), self.vae_temporal_compression_ratio
                        )
                        condition_tensor = condition_tensor[:, :, :valid_f]
                        print(f"[OmnimatteZero] Trimmed condition to {valid_f} frames for VAE compatibility")

                    conditioning_tensors.append(condition_tensor)
            else:
                # Raw tensors passed directly (from nodes.py)
                conditioning_tensors = conditions
                strength = [1.0]
                frame_index = [0]

        # 3. Prepare timesteps
        sigmas = linear_quadratic_schedule(num_inference_steps)
        timesteps = sigmas * 1000
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        latent_sigma = None
        if denoise_strength < 1:
            sigmas, timesteps, num_inference_steps = self.get_timesteps(
                sigmas, timesteps, num_inference_steps, denoise_strength
            )
            latent_sigma = sigmas[:1].repeat(batch_size)

        self._num_timesteps = len(timesteps)

        # 4. Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask, video_coords, extra_conditioning_num_latents = self.my_prepare_latents(
            conditioning_tensors, strength, frame_index,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height, width=width, num_frames=num_frames,
            sigma=latent_sigma, latents=latents,
            generator=generator, device=device, dtype=torch.float32,
        )

        video_coords = video_coords.float()
        video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)

        init_latents = latents.clone() if is_conditioning else None

        if self.do_classifier_free_guidance:
            video_coords = torch.cat([video_coords, video_coords], dim=0)

        # 5. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t

                if image_cond_noise_scale > 0 and init_latents is not None:
                    latents = self.add_noise_to_image_conditioning_latents(
                        t / 1000.0, init_latents, latents, image_cond_noise_scale,
                        conditioning_mask, generator,
                    )

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if is_conditioning and conditioning_mask is not None:
                    conditioning_mask_model_input = (
                        torch.cat([conditioning_mask, conditioning_mask])
                        if self.do_classifier_free_guidance else conditioning_mask
                    )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1).float()
                if is_conditioning and conditioning_mask is not None:
                    timestep = torch.min(timestep, (1 - conditioning_mask_model_input) * 1000.0)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    video_coords=video_coords,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    timestep, _ = timestep.chunk(2)

                denoised_latents = self.scheduler.step(
                    -noise_pred, t, latents, per_token_timesteps=timestep, return_dict=False
                )[0]

                if is_conditioning and conditioning_mask is not None:
                    tokens_to_denoise_mask = (
                        t / 1000 - 1e-6 < (1.0 - conditioning_mask)
                    ).unsqueeze(-1)
                    latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
                else:
                    latents = denoised_latents

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if is_conditioning:
            latents = latents[:, extra_conditioning_num_latents:]

        # 6. Decode
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        latents = self._unpack_latents(
            latents, latent_num_frames, latent_height, latent_width,
            self.transformer_spatial_patch_size, self.transformer_temporal_patch_size,
        )

        if output_type == "latent":
            return latents

        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor,
        )
        # Cast to VAE dtype (critical when VAE is float32 but transformer is bfloat16)
        latents = latents.to(self.vae.dtype)

        if not self.vae.config.timestep_conditioning:
            decode_ts = None
        else:
            # MPS: use safe randn for decode noise
            noise = _safe_randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size
            decode_ts = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
            dns = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[:, None, None, None, None]
            latents = (1 - dns) * latents + dns * noise

        video = self.vae.decode(latents, decode_ts, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        # MPS: flush the command buffer and free cached memory after decode
        _empty_cache_if_mps(device)

        self.maybe_free_model_hooks()

        if not isinstance(video, (list, tuple)):
            return video

        return LTXPipelineOutput(frames=video)


# ==============================================================================
# 2. FRAME ALIGNMENT UTILITY
# ==============================================================================

def fix_num_frames_for_vae(num_frames: int, temporal_compression_ratio: int = 8) -> int:
    """
    LTX VAE requires frame count of the form (k * ratio + 1).
    Valid counts: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, ...

    This is the ROOT CAUSE of the 'unflatten' error — aligning to
    multiples of 16 (e.g., 48) does NOT satisfy this constraint.
    """
    k = (num_frames - 1) // temporal_compression_ratio
    valid = k * temporal_compression_ratio + 1
    if valid < 1:
        valid = temporal_compression_ratio + 1
    return valid


# ==============================================================================
# 3. ATTENTION EXTRACTION
# ==============================================================================

class AttentionMapExtractor:
    def __init__(self, model: nn.Module, layer_indices: Optional[List[int]] = None,
                 attention_type: str = "self", average_over_heads: bool = True):
        self.model = model
        self.layer_indices = layer_indices
        self.attention_type = attention_type
        self.average_over_heads = average_over_heads
        self._hooks = []
        self._attention_maps = {}
        self._is_extracting = False

    def _find_attention_modules(self):
        attention_modules = []
        layer_idx = 0
        for name, module in self.model.named_modules():
            if 'attn1' in name or 'attn2' in name:
                is_attn1 = 'attn1' in name
                is_attn2 = 'attn2' in name
                if self.attention_type == "self" and not is_attn1:
                    continue
                if self.attention_type == "cross" and not is_attn2:
                    continue
                if self.layer_indices is None or layer_idx in self.layer_indices:
                    attention_modules.append((name, module, layer_idx))
                layer_idx += 1
        return attention_modules

    def _create_hook(self, layer_idx: int):
        stored_inputs = {}

        def pre_hook(module, args, kwargs):
            if not self._is_extracting:
                return
            if args:
                stored_inputs['hidden_states'] = args[0]
            if kwargs.get('hidden_states') is not None:
                stored_inputs['hidden_states'] = kwargs['hidden_states']
            if len(args) > 1:
                stored_inputs['encoder_hidden_states'] = args[1]

        def post_hook(module, args, kwargs, output):
            if not self._is_extracting:
                return
            try:
                hidden_states = stored_inputs.get('hidden_states')
                encoder_hidden_states = stored_inputs.get('encoder_hidden_states')
                query = module.to_q(hidden_states)
                key = module.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)

                batch_size, seq_len, _ = query.shape
                head_dim = query.shape[-1] // module.heads
                query = query.view(batch_size, seq_len, module.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

                # MPS: compute attention weights in float32 to avoid NaN from
                # half-precision softmax on large sequence lengths.  The
                # .detach().cpu() at the end means this doesn't affect model
                # memory — it's just for numerical stability during extraction.
                compute_dtype = query.dtype
                if _is_mps(query.device) and compute_dtype != torch.float32:
                    query = query.float()
                    key = key.float()

                scale = head_dim ** -0.5
                attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_weights, dim=-1)
                if self.average_over_heads:
                    attn_weights = attn_weights.mean(dim=1)
                self._attention_maps[layer_idx] = attn_weights.detach().cpu()
            except Exception:
                pass
            finally:
                stored_inputs.clear()

        return pre_hook, post_hook

    def register_hooks(self):
        self.remove_hooks()
        for name, module, idx in self._find_attention_modules():
            pre, post = self._create_hook(idx)
            self._hooks.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            self._hooks.append(module.register_forward_hook(post, with_kwargs=True))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @contextmanager
    def extraction_context(self):
        self._is_extracting = True
        self._attention_maps.clear()
        try:
            yield
        finally:
            self._is_extracting = False

    def get_maps(self):
        return self._attention_maps.copy()


# ==============================================================================
# 4. CUSTOM VAE (for foreground composition)
# ==============================================================================

class MyAutoencoderKLLTXVideo(AutoencoderKLLTXVideo):
    """
    Custom VAE with forward_composition for latent arithmetic.
    z_diff = z_all - z_bg; z_composed = z_new_bg + z_diff
    """

    def forward_composition(self, sample_list, generator=None):
        all_vid, bg, mask, mask2, new_bg = sample_list

        z_all = self.encode(all_vid).latent_dist.sample(generator=generator)
        z_bg = self.encode(bg).latent_dist.sample(generator=generator)
        z_new_bg = self.encode(new_bg).latent_dist.sample(generator=generator)

        z_diff = z_all - z_bg
        z_composed = z_new_bg + z_diff

        # MPS: flush after heavy VAE encode/decode sequence
        _empty_cache_if_mps(z_composed.device)

        return self.decode(z_composed, temb=None).sample
