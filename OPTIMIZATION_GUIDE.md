# Performance Optimization Guide for Chatterbox TTS on RTX 3090

## Problem Analysis

Based on your logs, the main performance issues were:

1. **Very low GPU memory utilization (9.15%)**: Only 2.2GB out of 24GB was being used
2. **RTF > 1**: Real-time factor greater than 1 means generation is slower than real-time
3. **Slow S3Gen waveform generation**: ~1.1s per prompt with 10 diffusion steps
4. **Chunked prefill enabled**: Adds overhead for small batch sizes
5. **Hardcoded diffusion steps**: Always using 10 steps regardless of quality needs

## Optimizations Applied

### 1. GPU Memory Utilization (Most Critical)

**Before:** Broken heuristic calculating only 9.15% GPU usage
```python
vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
vllm_memory_percent = vllm_memory_needed / unused_gpu_memory  # Result: 0.0915
```

**After:** Direct control with sensible default
```python
gpu_memory_utilization: float = 0.9  # Use 90% of GPU memory
```

**Impact:** More KV cache space → larger batches → better throughput

### 2. Diffusion Steps Reduction

**Before:** Hardcoded 10 steps in multiple places
- `tts.py`: `diffusion_steps: int = 10`
- `flow.py`: `n_timesteps=10`
- `s3gen.py`: `n_timesteps: int = 10`

**After:** Configurable with faster default
```python
diffusion_steps: int = 5  # 50% reduction, minimal quality loss
```

**Impact:** ~2x faster waveform generation (1.1s → ~0.55s)

### 3. vLLM Configuration Optimizations

Added these flags to `base_vllm_kwargs`:
```python
"disable_custom_all_reduce": True,      # Reduce overhead for single-GPU
"enable_chunked_prefill": False,        # Faster for small batches
"block_size": 16,                       # Efficient KV cache block size
```

### 4. Optional Model Compilation

Added support for `torch.compile`:
```python
if compile:
    s3gen.flow = torch.compile(s3gen.flow, mode="reduce-overhead")
```

Enable with `compile=True` parameter (adds initial compilation time, but speeds up subsequent runs).

## Usage Examples

### Basic Usage (Recommended for RTX 3090)

```python
model = ChatterboxTTS.from_pretrained(
    max_batch_size=3,
    max_model_len=1000,
    gpu_memory_utilization=0.9,  # 90% of 24GB = ~21.6GB
)

# Generate with default 5 diffusion steps (fast)
audios = model.generate(prompts, audio_prompt_path=None)
```

### Maximum Speed (Lower Quality)

```python
model = ChatterboxTTS.from_pretrained(
    gpu_memory_utilization=0.9,
    compile=True,  # Enable torch.compile
)

# Use only 3-4 diffusion steps for fastest generation
audios = model.generate(prompts, diffusion_steps=4)
```

### Higher Quality (Slower)

```python
model = ChatterboxTTS.from_pretrained(
    gpu_memory_utilization=0.9,
)

# Use 8-10 diffusion steps for best quality
audios = model.generate(prompts, diffusion_steps=8)
```

## Expected Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU Memory Usage | 9.15% (2.2GB) | 90% (21.6GB) | 10x more cache |
| S3Gen Time (per prompt) | ~1.1s | ~0.55s | 2x faster |
| T3 Generation | ~1.7s | ~1.5s | 15% faster |
| **Total RTF** | **>1.0** | **~0.5-0.6** | **~50% real-time** |

## Additional Tips

### 1. Tune GPU Memory Utilization

If you get OOM errors, reduce `gpu_memory_utilization`:
- RTX 3090 (24GB): Try 0.85-0.9
- RTX 4090 (24GB): Try 0.85-0.9
- RTX 3080 (10GB): Try 0.7-0.8

### 2. Batch Multiple Prompts

For better throughput, generate multiple prompts at once:
```python
prompts = ["text1", "text2", "text3", "text4"]
audios = model.generate(prompts, audio_prompt_path=None)
```

### 3. Adjust max_model_len

Shorter sequences need less memory:
```python
# For short phrases (<5 seconds)
model = ChatterboxTTS.from_pretrained(max_model_len=500)

# For longer content
model = ChatterboxTTS.from_pretrained(max_model_len=1500)
```

### 4. Monitor KV Cache Usage

Watch for this log line:
```
Maximum concurrency for 1,000 tokens per request: X.XXx
```
- If < 1.0: Increase `gpu_memory_utilization` or decrease `max_model_len`
- If > 2.0: You can increase `max_model_len` for longer generations

## Files Modified

1. `src/chatterbox_vllm/tts.py`: Main TTS interface and vLLM config
2. `src/chatterbox_vllm/models/s3gen/flow.py`: Flow matching inference
3. `src/chatterbox_vllm/models/s3gen/s3gen.py`: S3Gen wrapper
4. `example-tts.py`: Updated example with optimized settings

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce GPU memory utilization
model = ChatterboxTTS.from_pretrained(gpu_memory_utilization=0.7)

# Or reduce max_model_len
model = ChatterboxTTS.from_pretrained(max_model_len=500)
```

### Still Slow?

1. Ensure no other processes are using the GPU
2. Check thermal throttling (`nvidia-smi`)
3. Try enabling compilation: `compile=True`
4. Reduce diffusion steps: `diffusion_steps=3`

### Audio Quality Issues

If 5 diffusion steps produces poor quality:
```python
audios = model.generate(prompts, diffusion_steps=8)
```

## References

- [vLLM Performance Tuning](https://docs.vllm.ai/en/latest/performance.html)
- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [Flow Matching for TTS](https://arxiv.org/abs/2307.05463)
