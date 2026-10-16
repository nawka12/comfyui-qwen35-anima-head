import torch

import folder_paths

from .qwen35_anima import load_clip


class Qwen35AnimaCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip_name": (folder_paths.get_filename_list("text_encoders"), )},
                "optional": {"device": (["default", "cpu"], {"advanced": True})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Loads a Qwen3.5-0.8B + Anima head text encoder (qwen_3_5_08b_anima_head_L20 / imgloss files) for Anima. Use with the stock CLIPTextEncode node."

    def load(self, clip_name, device="default"):
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return (load_clip(path, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options=model_options),)


NODE_CLASS_MAPPINGS = {"Qwen35AnimaCLIPLoader": Qwen35AnimaCLIPLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen35AnimaCLIPLoader": "Load Qwen3.5 Anima Text Encoder"}
