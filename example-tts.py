#!/usr/bin/env python3

from typing import List
import torch
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    # Optimized settings for RTX 3090 (24GB)
    # - gpu_memory_utilization=0.75: Use 75% of GPU memory for vLLM (higher values may crash)
    # - compile=False: Disable torch.compile to avoid long compilation overhead for small batches
    # - s3gen_use_fp16=False: Keep FP32 for better quality (can enable for speed)
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
        gpu_memory_utilization=0.75,
        compile=False,  # Disable compilation for faster first-token latency
    )

    for i, audio_prompt_path in enumerate([None, "docs/audio-sample-01.mp3", "docs/audio-sample-03.mp3"]):
        prompts = [
            "You are listening to a demo of the Chatterbox TTS model running on VLLM.",
            "This is a separate prompt to test the batching implementation.",
            "And here is a third prompt. It's a bit longer than the first one, but not by much.",
        ]
    
        audios = model.generate(prompts, audio_prompt_path=audio_prompt_path, exaggeration=0.5)
        for audio_idx, audio in enumerate(audios):
            # Ensure audio is a tensor before saving
            if isinstance(audio, list):
                audio = torch.tensor(audio)
            ta.save(f"test-{i}-{audio_idx}.mp3", audio, model.sr)

    model.shutdown()