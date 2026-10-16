import types

import torch

import comfy.model_management
import comfy.sd
import comfy.sd1_clip
import comfy.text_encoders.anima as anima
import comfy.text_encoders.llama
import comfy.text_encoders.qwen35 as qwen35
import comfy.utils


class ExpRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        w = comfy.model_management.cast_to(self.weight, dtype=torch.float32, device=x.device)
        x32 = x.float()
        return (torch.nn.functional.rms_norm(x32, (x32.shape[-1],), eps=self.eps) * w.exp()).to(x.dtype)


class Qwen35HeadLM(torch.nn.Module):
    """Truncated Qwen3.5-0.8B whose final norm is replaced by the Anima head:
    Linear -> ExpRMSNorm -> SiLU -> Linear (keys norm.0 / norm.1 / norm.3)."""

    num_hidden_layers = 20

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        n = self.num_hidden_layers
        config = qwen35._make_config("qwen35_08b", {"num_hidden_layers": n, "layer_types": qwen35._qwen35_layer_types(n), "final_norm": False})
        self.num_layers = n
        self.model = qwen35.Qwen35Transformer(config, device=device, dtype=dtype, ops=operations)
        # fp32 norm weights and A_log: the file ships A_log and linear_attn.norm in
        # fp32, and ComfyUI's RMSNorm adds 1 to the weight in the weight's dtype,
        # which bf16 rounds (~1e-3 rel off diffucore/HF at the output).
        for mod in self.model.modules():
            if isinstance(mod, comfy.text_encoders.llama.RMSNorm):
                mod.weight.data = mod.weight.data.float()
            elif isinstance(mod, qwen35.GatedDeltaNet):
                mod.A_log.data = mod.A_log.data.float()
        h = config.hidden_size
        self.model.norm = torch.nn.Sequential(
            operations.Linear(h, 1024, bias=True, device=device, dtype=dtype),
            ExpRMSNorm(1024, eps=config.rms_norm_eps, device=device, dtype=dtype),
            torch.nn.SiLU(),
            operations.Linear(1024, 1024, bias=True, device=device, dtype=dtype),
        )
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)


class Qwen35HeadClipModel(comfy.sd1_clip.SDClipModel):
    lm_class = Qwen35HeadLM

    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 248044}, layer_norm_hidden_state=False, model_class=self.lm_class, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class Qwen35AnimaTokenizer(anima.AnimaTokenizer):
    # Same as stock Anima, with the Qwen3.5 vocab and no chat template. The
    # attribute keeps the stock name so AnimaTokenizer's methods apply as-is.
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.qwen3_06b = qwen35.Qwen35Tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, embedding_size=1024, embedding_key="qwen35_08b")
        self.t5xxl = anima.T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)


def te(num_layers, dtype_llama=None):
    class Qwen35HeadLM_(Qwen35HeadLM):
        num_hidden_layers = num_layers

    class Qwen35HeadClipModel_(Qwen35HeadClipModel):
        lm_class = Qwen35HeadLM_

    class Qwen35AnimaTEModel(anima.AnimaTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            comfy.sd1_clip.SD1ClipModel.__init__(self, device=device, dtype=dtype, name="qwen3_06b", clip_model=Qwen35HeadClipModel_, model_options=model_options)
    return Qwen35AnimaTEModel


def load_clip(ckpt_path, embedding_directory=None, model_options={}):
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    if "norm.0.weight" not in sd or "layers.0.linear_attn.A_log" not in sd:
        raise ValueError("{} is not a Qwen3.5 + Anima head text encoder (bare keys, norm.0/1/3 head)".format(ckpt_path))
    n = 0
    while "layers.{}.input_layernorm.weight".format(n) in sd:
        n += 1
    dtype = sd["layers.0.input_layernorm.weight"].dtype
    sd = {"model." + k: v for k, v in sd.items()}
    target = types.SimpleNamespace(params={}, clip=te(n, dtype_llama=dtype), tokenizer=Qwen35AnimaTokenizer)
    clip = comfy.sd.CLIP(target, embedding_directory=embedding_directory, parameters=comfy.utils.calculate_parameters(sd), state_dict=[sd], model_options=model_options)
    clip.patcher.cached_patcher_init = (load_clip_patcher, (ckpt_path, embedding_directory, model_options))
    return clip


def load_clip_patcher(ckpt_path, embedding_directory=None, model_options={}):
    return load_clip(ckpt_path, embedding_directory, model_options).patcher
