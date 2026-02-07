import torch
import numpy as np
import scipy.ndimage
import torch.nn.functional as F
from .omnimatte_impl import (
    OmnimatteZeroPipeline, MyAutoencoderKLLTXVideo,
    AttentionMapExtractor, fix_num_frames_for_vae
)
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput

# ==============================================================================
# Helpers
# ==============================================================================

def get_device(pipe):
    if torch.cuda.is_available():
        return torch.device("cuda")
    return pipe.device


def align_video_for_ltx_vae(video_tensor, vae_temporal_ratio=8, vae_spatial_ratio=32):
    """
    Align video tensor to LTX VAE requirements:
    - Frames: must be (k * vae_temporal_ratio + 1) → e.g. 9, 17, 25, 33, 41, 49, 57...
    - Spatial: must be divisible by vae_spatial_ratio (32)

    This is the FIX for the 'unflatten' crash — aligning to multiples of 16
    does NOT work because the VAE temporal downsampler requires (k*8+1) frames.
    """
    b, c, f, h, w = video_tensor.shape

    # Spatial: align to multiple of 32
    new_h = (h // vae_spatial_ratio) * vae_spatial_ratio
    new_w = (w // vae_spatial_ratio) * vae_spatial_ratio

    # Temporal: align to k*8+1 (the critical fix!)
    new_f = fix_num_frames_for_vae(f, vae_temporal_ratio)
    if new_f < vae_temporal_ratio + 1:
        new_f = vae_temporal_ratio + 1  # minimum 9 frames

    if new_f != f or new_h != h or new_w != w:
        print(f"[OmnimatteZero] Aligning: {f}f×{h}h×{w}w → {new_f}f×{new_h}h×{new_w}w")

    # Temporal: slice (don't interpolate time — preserves frame identity)
    if new_f != f:
        video_tensor = video_tensor[:, :, :new_f, :, :]

    # Spatial: resize if needed
    if new_h != h or new_w != w:
        video_tensor = F.interpolate(
            video_tensor, size=(new_f, new_h, new_w),
            mode='trilinear', align_corners=False
        )

    return video_tensor


def comfy_to_diffusers(image):
    """Comfy [F, H, W, C] 0..1 → Diffusers [1, C, F, H, W] -1..1"""
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    video = image.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
    return video * 2.0 - 1.0


def diffusers_to_comfy(video):
    """Diffusers [1, C, F, H, W] → Comfy [F, H, W, C] 0..1"""
    video = video.squeeze(0).permute(1, 2, 3, 0)
    return (video * 0.5 + 0.5).clamp(0, 1)


def align_mask_to_video(mask_tensor, target_f, target_h, target_w):
    """
    Ensure mask matches video dimensions exactly.
    mask_tensor: (F, H, W) or (1, 1, F, H, W)
    """
    if mask_tensor.dim() == 3:
        # (F, H, W) → (1, 1, F, H, W)
        mask_5d = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.dim() == 5:
        mask_5d = mask_tensor
    else:
        mask_5d = mask_tensor.view(1, 1, -1, mask_tensor.shape[-2], mask_tensor.shape[-1])

    curr_f = mask_5d.shape[2]

    # Temporal alignment
    if curr_f > target_f:
        mask_5d = mask_5d[:, :, :target_f, :, :]
    elif curr_f < target_f:
        pad = target_f - curr_f
        last = mask_5d[:, :, -1:, :, :]
        mask_5d = torch.cat([mask_5d, last.repeat(1, 1, pad, 1, 1)], dim=2)

    # Spatial alignment
    if mask_5d.shape[-2] != target_h or mask_5d.shape[-1] != target_w:
        mask_5d = F.interpolate(mask_5d, size=(target_f, target_h, target_w), mode='nearest')

    return mask_5d


# ==============================================================================
# Node: Loader
# ==============================================================================

class OmnimatteLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id": ("STRING", {"default": "a-r-r-o-w/LTX-Video-0.9.7-diffusers"}),
                "precision": (["bf16", "fp16"], {"default": "bf16"}),
                "vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable VAE tiling for lower VRAM. May cause artifacts on some frame counts."
                }),
                "vae_slicing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable VAE slicing to process frames sequentially (saves VRAM)."
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable model CPU offload to reduce VRAM usage."
                }),
                "vae_float32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force VAE to float32 (avoids slow_conv3d errors on some GPUs)."
                }),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("OMNIMATTE_PIPE", "VAE")
    FUNCTION = "load"
    CATEGORY = "OmnimatteZero"

    def load(self, model_id, precision, vae_tiling, vae_slicing, cpu_offload, vae_float32, cache_dir=""):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        cache = cache_dir if cache_dir else None

        print(f"[OmnimatteZero] Loading pipeline: {model_id} ({precision})")

        pipe = OmnimatteZeroPipeline.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=cache
        )

        # Inject Custom VAE for composition support
        custom_vae = MyAutoencoderKLLTXVideo.from_config(pipe.vae.config)
        custom_vae.load_state_dict(pipe.vae.state_dict())

        if vae_float32:
            custom_vae.to(dtype=torch.float32).eval()
            print("[OmnimatteZero] VAE forced to float32")
        else:
            custom_vae.to(dtype=dtype).eval()

        if vae_tiling:
            custom_vae.enable_tiling()
            print("[OmnimatteZero] VAE tiling enabled")

        if vae_slicing:
            custom_vae.enable_slicing()
            print("[OmnimatteZero] VAE slicing enabled")

        pipe.vae = custom_vae

        if cpu_offload:
            pipe.enable_model_cpu_offload()
            print("[OmnimatteZero] CPU offload enabled")
        else:
            pipe.to("cuda")
            print("[OmnimatteZero] All models on GPU")

        print("[OmnimatteZero] Pipeline ready")
        return (pipe, custom_vae)


# ==============================================================================
# Node: Total Mask Generator (Self-Attention)
# ==============================================================================

class OmnimatteTotalMaskGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("OMNIMATTE_PIPE",),
                "video": ("IMAGE",),
                "object_mask": ("MASK",),
                "dilation": ("INT", {"default": 5, "min": 0, "max": 20}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("total_mask",)
    FUNCTION = "generate"
    CATEGORY = "OmnimatteZero"

    def generate(self, pipe, video, object_mask, dilation):
        device = get_device(pipe)

        # Align video for VAE
        vid_input = comfy_to_diffusers(video).to(device, dtype=torch.float32)
        vid_input = align_video_for_ltx_vae(
            vid_input,
            vae_temporal_ratio=pipe.vae_temporal_compression_ratio,
            vae_spatial_ratio=pipe.vae_spatial_compression_ratio,
        )

        target_f = vid_input.shape[2]
        target_h = vid_input.shape[3]
        target_w = vid_input.shape[4]

        # Hook attention
        extractor = AttentionMapExtractor(pipe.transformer, attention_type="self")
        extractor.register_hooks()

        print(f"[OmnimatteZero] Extracting attention maps from video: {vid_input.shape}")
        dtype = pipe.transformer.dtype

        with torch.no_grad(), extractor.extraction_context():
            latents = pipe.vae.encode(vid_input).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            latents = latents.to(dtype=dtype)

            bs = latents.shape[0]
            t = torch.tensor([500], device=device)
            p_emb, p_mask, _, _ = pipe.encode_prompt("", device=device)

            h_lat, w_lat = latents.shape[-2], latents.shape[-1]
            vid_ids = pipe._prepare_video_ids(
                bs, latents.shape[2], h_lat, w_lat,
                pipe.transformer_temporal_patch_size,
                pipe.transformer_spatial_patch_size,
                device,
            )
            vid_ids = pipe._scale_video_ids(
                vid_ids,
                pipe.vae_spatial_compression_ratio,
                pipe.vae_temporal_compression_ratio,
                0, device,
            )

            packed = pipe._pack_latents(
                latents,
                pipe.transformer_spatial_patch_size,
                pipe.transformer_temporal_patch_size,
            )
            pipe.transformer(
                packed, p_emb, t,
                encoder_attention_mask=p_mask,
                video_coords=vid_ids,
            )

        extractor.remove_hooks()

        # Process mask
        mask_np = object_mask.cpu().numpy()

        # Align mask frame count
        if mask_np.shape[0] > target_f:
            mask_np = mask_np[:target_f]
        elif mask_np.shape[0] < target_f:
            pad = target_f - mask_np.shape[0]
            mask_np = np.concatenate([mask_np, np.repeat(mask_np[-1:], pad, axis=0)], axis=0)

        # Dilate
        result_frames = []
        struct = np.ones((dilation, dilation)) if dilation > 0 else None
        for i in range(mask_np.shape[0]):
            m = mask_np[i]
            if struct is not None:
                d = scipy.ndimage.binary_dilation(m > 0.5, structure=struct)
            else:
                d = m > 0.5
            result_frames.append(d)

        final_mask = np.stack(result_frames)
        mask_tensor = torch.from_numpy(final_mask).float()

        # Resize to match aligned video
        if mask_tensor.shape[-2:] != (target_h, target_w):
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(target_f, target_h, target_w),
                mode='nearest',
            ).squeeze(0).squeeze(0)

        return (mask_tensor,)


# ==============================================================================
# Node: Object Removal
# ==============================================================================

class OmnimatteObjectRemoval:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("OMNIMATTE_PIPE",),
                "video": ("IMAGE",),
                "total_mask": ("MASK",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {
                    "default": "worst quality, inconsistent motion, blurry, jittery, distorted"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("background_video",)
    FUNCTION = "remove"
    CATEGORY = "OmnimatteZero"

    def remove(self, pipe, video, total_mask, steps, guidance_scale=3.0, seed=42,
               prompt="", negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted"):
        device = get_device(pipe)
        vae_dtype = pipe.vae.dtype

        # 1. Prepare video — align to k*8+1 frames
        vid_in = comfy_to_diffusers(video).to(device, dtype=vae_dtype)
        vid_in = align_video_for_ltx_vae(
            vid_in,
            vae_temporal_ratio=pipe.vae_temporal_compression_ratio,
            vae_spatial_ratio=pipe.vae_spatial_compression_ratio,
        )

        target_f = vid_in.shape[2]
        target_h = vid_in.shape[3]
        target_w = vid_in.shape[4]

        print(f"[OmnimatteZero] Object removal: {target_f} frames @ {target_w}x{target_h}, {steps} steps")

        # 2. Prepare mask — align to same dimensions
        mask_in = total_mask.to(device, dtype=vae_dtype)
        mask_5d = align_mask_to_video(mask_in, target_f, target_h, target_w)

        # Convert mask to 3-channel (RGB) like the original code expects
        # mask needs shape [1, 3, F, H, W] for VAE encoding
        mask_3ch = mask_5d.expand(-1, 3, -1, -1, -1)
        # Scale to [-1, 1] range (diffusers video format)
        mask_3ch = mask_3ch * 2.0 - 1.0

        # 3. Run pipeline (use PIL output — reliable format)
        output = pipe.my_call(
            conditions=[vid_in, mask_3ch],
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=target_f,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="pil",
        )

        # 4. Convert PIL output to ComfyUI IMAGE format (F, H, W, C) float [0,1]
        if isinstance(output, LTXPipelineOutput):
            pil_frames = output.frames[0]  # list of PIL images
        elif isinstance(output, (list, tuple)):
            pil_frames = output[0] if isinstance(output[0], list) else output
        else:
            pil_frames = output

        frames = []
        for img in pil_frames:
            frames.append(np.array(img).astype(np.float32) / 255.0)
        result_tensor = torch.from_numpy(np.stack(frames))

        return (result_tensor,)


# ==============================================================================
# Node: Composition
# ==============================================================================

class OmnimatteComposition:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "orig_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "new_background": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composed_video",)
    FUNCTION = "compose"
    CATEGORY = "OmnimatteZero"

    def compose(self, vae, orig_video, background_video, new_background):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float32

        vae_t_ratio = getattr(vae, 'config', {})
        temporal_ratio = 8  # LTX default
        spatial_ratio = 32

        def prep(v):
            x = comfy_to_diffusers(v).to(device, dtype=dtype)
            return align_video_for_ltx_vae(x, temporal_ratio, spatial_ratio)

        v_orig = prep(orig_video)
        v_bg = prep(background_video)
        v_new = prep(new_background)

        # Match frame counts
        min_f = min(v_orig.shape[2], v_bg.shape[2], v_new.shape[2])
        # Ensure min_f is also VAE-compatible
        min_f = fix_num_frames_for_vae(min_f, temporal_ratio)
        v_orig = v_orig[:, :, :min_f]
        v_bg = v_bg[:, :, :min_f]
        v_new = v_new[:, :, :min_f]

        sample_list = [v_orig, v_bg, None, None, v_new]

        with torch.no_grad():
            out = vae.forward_composition(sample_list)

        return (diffusers_to_comfy(out.cpu()),)


# ==============================================================================
# Node Mappings
# ==============================================================================

# ==============================================================================
# Node Mappings
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "OmnimatteLoader": OmnimatteLoader,
    "OmnimatteTotalMaskGen": OmnimatteTotalMaskGen,
    "OmnimatteObjectRemoval": OmnimatteObjectRemoval,
    "OmnimatteComposition": OmnimatteComposition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmnimatteLoader": "🎬 Omnimatte Loader",
    "OmnimatteTotalMaskGen": "🔍 Omnimatte Mask Gen",
    "OmnimatteObjectRemoval": "🧹 Omnimatte Object Removal",
    "OmnimatteComposition": "✂️ Omnimatte Composition",
}
