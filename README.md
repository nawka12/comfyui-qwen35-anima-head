# Qwen3.5 Anima text encoder for ComfyUI

A loader node that lets Anima use the Qwen3.5-0.8B + head text encoder
(`qwen_3_5_08b_imgloss_*.safetensors`, `qwen_3_5_08b_anima_head_L20.safetensors`)
in place of Qwen3-0.6B. ComfyUI's built-in loader can't read these files: they use
bare keys, 20 layers instead of 24, and a projection head in place of the final norm.

## Install

You need a ComfyUI recent enough to ship `comfy/text_encoders/qwen35.py`
(July 2026 or newer). The node was tested on commits `f8a3fd9d` (2026-07-24) and
`9fa63b0a` (2026-10-15). It has no extra Python dependencies.

1. Install the node, either with git:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/nawka12/comfyui-qwen35-anima-head
   ```

   or by downloading the ZIP from GitHub (**Code** > **Download ZIP**) and
   extracting it into `ComfyUI/custom_nodes/`.

2. Download the text encoder, `qwen_3_5_08b_anima_head_L20.safetensors`, from
   [Civitai](https://civitai.com/models/3003960) and put it in
   `ComfyUI/models/text_encoders/`.

3. Restart ComfyUI. The node shows up as **Load Qwen3.5 Anima Text Encoder**
   under model/loaders.

To update later, run `git pull` inside `ComfyUI/custom_nodes/comfyui-qwen35-anima-head`.

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

## License

The node is GPL-3.0, the same as ComfyUI, which it builds on. The text encoder
weights are licensed separately: the projection head is under the CircleStone
Labs Non-Commercial License v1.2 and the Qwen3.5-0.8B layers are Apache-2.0. See
the Civitai page for details.
