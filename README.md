# Qwen3.5 Anima text encoder for ComfyUI

A loader node that lets Anima use the Qwen3.5-0.8B + head text encoder
(`qwen_3_5_08b_imgloss_*.safetensors`, `qwen_3_5_08b_anima_head_L20.safetensors`)
in place of Qwen3-0.6B. ComfyUI's built-in loader can't read these files: they use
bare keys, 20 layers instead of 24, and a projection head in place of the final norm.

## Install

1. Copy this folder into `ComfyUI/custom_nodes/` (any folder name works).
2. Put the `.safetensors` file in `ComfyUI/models/text_encoders/`.
3. Restart ComfyUI.

This needs a ComfyUI version that ships `comfy/text_encoders/qwen35.py`. It was
tested against commits `f8a3fd9d` (2026-07-24) and `9fa63b0a` (2026-10-15).

## Use

**Load Qwen3.5 Anima Text Encoder** (model/loaders) -> CLIP -> the stock
**CLIP Text Encode** nodes -> your Anima model. Everything else in an Anima
workflow stays as it is. Prompts are handled the same way as with the stock
Anima text encoder, so `(tag:1.2)` weights and `\(` escapes work as usual.

## Notes

- The node reuses ComfyUI's own Qwen3.5 layers. Its output matches diffucore's
  encoder to about 2e-5 max relative error at fp32.
- Like other ComfyUI text encoders, it runs in fp32. The norm weights stay in
  fp32 too, because ComfyUI's RMSNorm rounds `1 + weight` to bf16 otherwise.
- An empty prompt becomes a single `<|endoftext|>` token, the same convention
  as stock Anima in ComfyUI. Long prompts are not truncated (diffucore cuts at
  512 tokens).
