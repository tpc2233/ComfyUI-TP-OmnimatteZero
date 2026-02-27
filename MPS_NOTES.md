# Apple Silicon (MPS) Support

## Overview

This fork adds Apple Silicon support via PyTorch's MPS (Metal Performance Shaders) backend. The nodes auto-detect the available device — CUDA, MPS, or CPU — so the same code runs on both NVIDIA and Apple hardware with no configuration changes.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- PyTorch >= 2.1 (tested on 2.10.0)
- macOS 15+ recommended
- `diffusers >= 0.32.1`

## Recommended Loader Settings for MPS

| Setting | Recommended | Notes |
|---------|------------|-------|
| **precision** | `fp16` | **Required.** bf16 passes basic tests but produces visual artifacts (hallucinations, corrupted inpainting) due to accumulated rounding errors across the denoising loop. The code auto-downgrades bf16 → fp16 on MPS. |
| **vae_float32** | `True` | **Strongly recommended.** The LTX VAE's Conv3d layers can produce NaN in half precision on MPS. |
| **cpu_offload** | Either | Ignored on MPS. Apple's unified memory architecture means there is no separate VRAM to offload from — the setting has no effect. |
| **vae_tiling** | `False` | Use if needed for very high resolution. |
| **vae_slicing** | `False` | Use if needed for long sequences. |

## VRAM / Memory

The LTX-Video-0.9.7 model requires approximately 90GB of memory. This exceeds the VRAM of all current consumer NVIDIA GPUs but fits comfortably in the unified memory of Apple Silicon Macs with 96GB+ RAM (M4 Pro/Max/Ultra configurations).

This is one of the primary use cases for the MPS port — models that are too large for discrete GPU VRAM can run on high-memory Apple Silicon machines.

## What's Different on MPS

### Automatic changes (handled in code)
- **Device detection**: CUDA → MPS → CPU cascade, no user action needed
- **bf16 → fp16 downgrade**: Automatic when MPS is detected
- **cpu_offload bypass**: Skipped on MPS (unified memory makes it unnecessary and the diffusers implementation uses CUDA-specific APIs)
- **Generator device**: Random number generation uses CPU generator with tensors moved to MPS, avoiding MPS Generator reliability issues in some PyTorch versions
- **Attention extraction**: Float32 upcast for softmax stability on large sequence lengths
- **Memory management**: `torch.mps.synchronize()` + `torch.mps.empty_cache()` after heavy operations

### Not available on MPS
- xformers (CUDA-only)
- Flash Attention (CUDA-only)
- SageAttention (CUDA-only)
- Triton kernels (CUDA-only)

PyTorch's built-in `scaled_dot_product_attention` handles attention on MPS using Metal shaders.

## Known Limitations

- **No bf16**: Must use fp16. The bf16 precision format produces cumulative errors during the 30+ step denoising process that manifest as visual artifacts in the output.
- **No CPU offload**: The setting is accepted but ignored. This is not a limitation in practice since unified memory eliminates the need for offloading.
- **Speed**: MPS inference is slower than equivalent CUDA hardware. There is no Flash Attention or kernel fusion available. `torch.compile()` with MPS backend is an area of active investigation.
- **Model quality**: Object removal quality is determined by the LTX-Video model, not the compute backend. Results on MPS match fp16 CUDA results. The v0.9.7 model runs without explicit temporal/spatial attention guidance (TAG/SAG), which limits inpainting quality on complex scenes (e.g., large object removal with textured backgrounds, moving cameras, urban environments).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Visual artifacts / hallucinations | bf16 precision | Switch to fp16 (should happen automatically) |
| NaN in output / black frames | VAE running in half precision | Enable `vae_float32` |
| `Torch not compiled with CUDA enabled` | Old version of nodes without MPS patch | Update to this fork |
| `enable_model_cpu_offload` error | Diffusers calling CUDA memory APIs | Should not occur with patched code; cpu_offload is bypassed on MPS |
| Slow performance | Normal for MPS vs CUDA | No fix currently; torch.compile may help in future |

## Tested On

- Mac Studio M4 Max, 128GB unified memory
- macOS 15.7.4
- PyTorch 2.10.0
- ComfyUI 0.12.2
- diffusers 0.32.x
