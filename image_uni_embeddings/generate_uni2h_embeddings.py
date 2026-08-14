# We ran this on the server (to benefit from GPU speedups).
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pickle

samples_prostate = ["PCA007", "PCA008", "PCA009", "PCA010"]
samples_lung = ["VUILD102MA", "VUILD142MA", "VUILD48LA1", "THD0008", "VUHD069", "VUILD106MA", "TILD113LA", "VUILD78MA", "VUHD113", "VUILD91MA", "TILD117LA", "VUHD116A", "VUILD110LA", "TILD299MA", "VUILD141MA", "TILD315MA", "VUILD104MA1", "VUILD115MA", "VUILD96MA", "VUILD105MA1", "TILD175MA", "VUHD038", "VUHD090", "VUILD96LA", "VUILD48LA2", "TILD111LA", "TILD130LA", "VUHD116B", "TILD080LA", "VUILD91LA", "VUILD104MA2", "VUILD78LA", "VUILD107MA", "VUILD105MA2", "THD0011", "VUILD102LA", "VUILD49LA", "VUILD58MA", "TILD049MA"]

uni_processing_dict= {
    "prostate": {"samples": samples_prostate, "patch_size": "224", "input_dir": "prostate_input", "output_dir": "uni_prostate_output"},
    "lung": {"samples": samples_lung, "patch_size": "112", "input_dir": "lung_input", "output_dir": "uni_lung_output"},
}

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)

    # loop through the datasets and patches
    for dataset_key in uni_processing_dict.keys():
        for sample in uni_processing_dict[dataset_key]['samples']:
            print(f"Dataset: {dataset_key}. Processing sample: {sample}")
            patch_dir = f"{uni_processing_dict[dataset_key]['input_dir']}/he_size_{uni_processing_dict[dataset_key]['patch_size']}_microns_patches_{sample}"
            embeddings = {}

            for fname in tqdm(os.listdir(patch_dir)):
                if not fname.lower().endswith(".png"):
                    continue
                if fname.lower().startswith("._"):
                    continue
                path = os.path.join(patch_dir, fname)
                img = Image.open(path).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(device)
                with torch.inference_mode():
                    emb = model(img_t)
                emb = emb.cpu().numpy().reshape(-1)
                spot_id = os.path.splitext(fname)[0]
                embeddings[spot_id] = emb

            with open(f"{uni_processing_dict[dataset_key]['output_dir']}/uni2h_{sample}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

            print("Done. Extracted embeddings for {} patches.".format(len(embeddings)))
