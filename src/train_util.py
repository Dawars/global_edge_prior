import numpy as np
import re
import torch
import yaml
from pathlib import Path
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

class ClearCacheCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            print(f"Epoch {trainer.current_epoch}: CUDA cache cleared")

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            
def frozen_layer_names(num_train_layers=4):
    if num_train_layers==4:
        return [
"frozen_encoder.backbone.model.cls_token", 
"frozen_encoder.backbone.model.pos_embed", 
"frozen_encoder.backbone.model.mask_token", 
"frozen_encoder.backbone.model.patch_embed.proj.weight", 
"frozen_encoder.backbone.model.patch_embed.proj.bias", 
"frozen_encoder.backbone.model.blocks.0.norm1.weight", 
"frozen_encoder.backbone.model.blocks.0.norm1.bias", 
"frozen_encoder.backbone.model.blocks.0.attn.qkv.weight", 
"frozen_encoder.backbone.model.blocks.0.attn.qkv.bias", 
"frozen_encoder.backbone.model.blocks.0.attn.proj.weight", 
"frozen_encoder.backbone.model.blocks.0.attn.proj.bias", 
"frozen_encoder.backbone.model.blocks.0.ls1.gamma", 
"frozen_encoder.backbone.model.blocks.0.norm2.weight", 
"frozen_encoder.backbone.model.blocks.0.norm2.bias", 
"frozen_encoder.backbone.model.blocks.0.mlp.fc1.weight", 
"frozen_encoder.backbone.model.blocks.0.mlp.fc1.bias", 
"frozen_encoder.backbone.model.blocks.0.mlp.fc2.weight", 
"frozen_encoder.backbone.model.blocks.0.mlp.fc2.bias", 
"frozen_encoder.backbone.model.blocks.0.ls2.gamma", 
"frozen_encoder.backbone.model.blocks.1.norm1.weight", 
"frozen_encoder.backbone.model.blocks.1.norm1.bias", 
"frozen_encoder.backbone.model.blocks.1.attn.qkv.weight", 
"frozen_encoder.backbone.model.blocks.1.attn.qkv.bias", 
"frozen_encoder.backbone.model.blocks.1.attn.proj.weight", 
"frozen_encoder.backbone.model.blocks.1.attn.proj.bias", 
"frozen_encoder.backbone.model.blocks.1.ls1.gamma", 
"frozen_encoder.backbone.model.blocks.1.norm2.weight", 
"frozen_encoder.backbone.model.blocks.1.norm2.bias", 
"frozen_encoder.backbone.model.blocks.1.mlp.fc1.weight", 
"frozen_encoder.backbone.model.blocks.1.mlp.fc1.bias", 
"frozen_encoder.backbone.model.blocks.1.mlp.fc2.weight", 
"frozen_encoder.backbone.model.blocks.1.mlp.fc2.bias", 
"frozen_encoder.backbone.model.blocks.1.ls2.gamma", 
"frozen_encoder.backbone.model.blocks.2.norm1.weight", 
"frozen_encoder.backbone.model.blocks.2.norm1.bias", 
"frozen_encoder.backbone.model.blocks.2.attn.qkv.weight", 
"frozen_encoder.backbone.model.blocks.2.attn.qkv.bias", 
"frozen_encoder.backbone.model.blocks.2.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.2.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.2.ls1.gamma","frozen_encoder.backbone.model.blocks.2.norm2.weight",
"frozen_encoder.backbone.model.blocks.2.norm2.bias",
"frozen_encoder.backbone.model.blocks.2.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.2.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.2.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.2.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.2.ls2.gamma","frozen_encoder.backbone.model.blocks.3.norm1.weight",
"frozen_encoder.backbone.model.blocks.3.norm1.bias",
"frozen_encoder.backbone.model.blocks.3.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.3.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.3.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.3.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.3.ls1.gamma","frozen_encoder.backbone.model.blocks.3.norm2.weight",
"frozen_encoder.backbone.model.blocks.3.norm2.bias",
"frozen_encoder.backbone.model.blocks.3.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.3.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.3.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.3.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.3.ls2.gamma","frozen_encoder.backbone.model.blocks.4.norm1.weight",
"frozen_encoder.backbone.model.blocks.4.norm1.bias",
"frozen_encoder.backbone.model.blocks.4.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.4.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.4.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.4.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.4.ls1.gamma","frozen_encoder.backbone.model.blocks.4.norm2.weight",
"frozen_encoder.backbone.model.blocks.4.norm2.bias",
"frozen_encoder.backbone.model.blocks.4.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.4.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.4.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.4.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.4.ls2.gamma","frozen_encoder.backbone.model.blocks.5.norm1.weight",
"frozen_encoder.backbone.model.blocks.5.norm1.bias",
"frozen_encoder.backbone.model.blocks.5.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.5.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.5.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.5.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.5.ls1.gamma",
"frozen_encoder.backbone.model.blocks.5.norm2.weight",
"frozen_encoder.backbone.model.blocks.5.norm2.bias",
"frozen_encoder.backbone.model.blocks.5.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.5.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.5.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.5.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.5.ls2.gamma",
"frozen_encoder.backbone.model.blocks.6.norm1.weight",
"frozen_encoder.backbone.model.blocks.6.norm1.bias",
"frozen_encoder.backbone.model.blocks.6.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.6.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.6.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.6.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.6.ls1.gamma",
"frozen_encoder.backbone.model.blocks.6.norm2.weight",
"frozen_encoder.backbone.model.blocks.6.norm2.bias",
"frozen_encoder.backbone.model.blocks.6.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.6.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.6.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.6.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.6.ls2.gamma",
"frozen_encoder.backbone.model.blocks.7.norm1.weight",
"frozen_encoder.backbone.model.blocks.7.norm1.bias",
"frozen_encoder.backbone.model.blocks.7.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.7.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.7.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.7.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.7.ls1.gamma",
"frozen_encoder.backbone.model.blocks.7.norm2.weight",
"frozen_encoder.backbone.model.blocks.7.norm2.bias",
"frozen_encoder.backbone.model.blocks.7.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.7.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.7.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.7.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.7.ls2.gamma"]

    else:
        return [
"frozen_encoder.backbone.model.cls_token", 
"frozen_encoder.backbone.model.pos_embed", 
"frozen_encoder.backbone.model.mask_token", 
"frozen_encoder.backbone.model.patch_embed.proj.weight", 
"frozen_encoder.backbone.model.patch_embed.proj.bias", 
"frozen_encoder.backbone.model.blocks.0.norm1.weight", 
"frozen_encoder.backbone.model.blocks.0.norm1.bias", 
"frozen_encoder.backbone.model.blocks.0.attn.qkv.weight", 
"frozen_encoder.backbone.model.blocks.0.attn.qkv.bias", 
"frozen_encoder.backbone.model.blocks.0.attn.proj.weight", 
"frozen_encoder.backbone.model.blocks.0.attn.proj.bias", 
"frozen_encoder.backbone.model.blocks.0.ls1.gamma", 
"frozen_encoder.backbone.model.blocks.0.norm2.weight", 
"frozen_encoder.backbone.model.blocks.0.norm2.bias", 
"frozen_encoder.backbone.model.blocks.0.mlp.fc1.weight", 
"frozen_encoder.backbone.model.blocks.0.mlp.fc1.bias", 
"frozen_encoder.backbone.model.blocks.0.mlp.fc2.weight", 
"frozen_encoder.backbone.model.blocks.0.mlp.fc2.bias", 
"frozen_encoder.backbone.model.blocks.0.ls2.gamma", 
"frozen_encoder.backbone.model.blocks.1.norm1.weight", 
"frozen_encoder.backbone.model.blocks.1.norm1.bias", 
"frozen_encoder.backbone.model.blocks.1.attn.qkv.weight", 
"frozen_encoder.backbone.model.blocks.1.attn.qkv.bias", 
"frozen_encoder.backbone.model.blocks.1.attn.proj.weight", 
"frozen_encoder.backbone.model.blocks.1.attn.proj.bias", 
"frozen_encoder.backbone.model.blocks.1.ls1.gamma", 
"frozen_encoder.backbone.model.blocks.1.norm2.weight", 
"frozen_encoder.backbone.model.blocks.1.norm2.bias", 
"frozen_encoder.backbone.model.blocks.1.mlp.fc1.weight", 
"frozen_encoder.backbone.model.blocks.1.mlp.fc1.bias", 
"frozen_encoder.backbone.model.blocks.1.mlp.fc2.weight", 
"frozen_encoder.backbone.model.blocks.1.mlp.fc2.bias", 
"frozen_encoder.backbone.model.blocks.1.ls2.gamma", 
"frozen_encoder.backbone.model.blocks.2.norm1.weight", 
"frozen_encoder.backbone.model.blocks.2.norm1.bias", 
"frozen_encoder.backbone.model.blocks.2.attn.qkv.weight", 
"frozen_encoder.backbone.model.blocks.2.attn.qkv.bias", 
"frozen_encoder.backbone.model.blocks.2.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.2.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.2.ls1.gamma","frozen_encoder.backbone.model.blocks.2.norm2.weight",
"frozen_encoder.backbone.model.blocks.2.norm2.bias",
"frozen_encoder.backbone.model.blocks.2.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.2.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.2.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.2.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.2.ls2.gamma","frozen_encoder.backbone.model.blocks.3.norm1.weight",
"frozen_encoder.backbone.model.blocks.3.norm1.bias",
"frozen_encoder.backbone.model.blocks.3.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.3.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.3.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.3.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.3.ls1.gamma","frozen_encoder.backbone.model.blocks.3.norm2.weight",
"frozen_encoder.backbone.model.blocks.3.norm2.bias",
"frozen_encoder.backbone.model.blocks.3.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.3.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.3.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.3.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.3.ls2.gamma","frozen_encoder.backbone.model.blocks.4.norm1.weight",
"frozen_encoder.backbone.model.blocks.4.norm1.bias",
"frozen_encoder.backbone.model.blocks.4.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.4.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.4.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.4.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.4.ls1.gamma","frozen_encoder.backbone.model.blocks.4.norm2.weight",
"frozen_encoder.backbone.model.blocks.4.norm2.bias",
"frozen_encoder.backbone.model.blocks.4.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.4.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.4.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.4.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.4.ls2.gamma","frozen_encoder.backbone.model.blocks.5.norm1.weight",
"frozen_encoder.backbone.model.blocks.5.norm1.bias",
"frozen_encoder.backbone.model.blocks.5.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.5.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.5.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.5.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.5.ls1.gamma",
"frozen_encoder.backbone.model.blocks.5.norm2.weight",
"frozen_encoder.backbone.model.blocks.5.norm2.bias",
"frozen_encoder.backbone.model.blocks.5.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.5.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.5.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.5.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.5.ls2.gamma",
"frozen_encoder.backbone.model.blocks.6.norm1.weight",
"frozen_encoder.backbone.model.blocks.6.norm1.bias",
"frozen_encoder.backbone.model.blocks.6.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.6.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.6.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.6.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.6.ls1.gamma",
"frozen_encoder.backbone.model.blocks.6.norm2.weight",
"frozen_encoder.backbone.model.blocks.6.norm2.bias",
"frozen_encoder.backbone.model.blocks.6.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.6.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.6.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.6.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.6.ls2.gamma",
"frozen_encoder.backbone.model.blocks.7.norm1.weight",
"frozen_encoder.backbone.model.blocks.7.norm1.bias",
"frozen_encoder.backbone.model.blocks.7.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.7.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.7.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.7.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.7.ls1.gamma",
"frozen_encoder.backbone.model.blocks.7.norm2.weight",
"frozen_encoder.backbone.model.blocks.7.norm2.bias",
"frozen_encoder.backbone.model.blocks.7.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.7.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.7.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.7.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.7.ls2.gamma",
"frozen_encoder.backbone.model.blocks.8.norm1.weight",
"frozen_encoder.backbone.model.blocks.8.norm1.bias",
"frozen_encoder.backbone.model.blocks.8.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.8.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.8.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.8.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.8.ls1.gamma",
"frozen_encoder.backbone.model.blocks.8.norm2.weight",
"frozen_encoder.backbone.model.blocks.8.norm2.bias",
"frozen_encoder.backbone.model.blocks.8.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.8.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.8.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.8.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.8.ls2.gamma",
"frozen_encoder.backbone.model.blocks.9.norm1.weight",
"frozen_encoder.backbone.model.blocks.9.norm1.bias",
"frozen_encoder.backbone.model.blocks.9.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.9.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.9.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.9.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.9.ls1.gamma",
"frozen_encoder.backbone.model.blocks.9.norm2.weight",
"frozen_encoder.backbone.model.blocks.9.norm2.bias",
"frozen_encoder.backbone.model.blocks.9.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.9.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.9.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.9.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.9.ls2.gamma",
"frozen_encoder.backbone.model.blocks.10.norm1.weight",
"frozen_encoder.backbone.model.blocks.10.norm1.bias",
"frozen_encoder.backbone.model.blocks.10.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.10.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.10.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.10.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.10.ls1.gamma",
"frozen_encoder.backbone.model.blocks.10.norm2.weight",
"frozen_encoder.backbone.model.blocks.10.norm2.bias",
"frozen_encoder.backbone.model.blocks.10.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.10.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.10.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.10.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.10.ls2.gamma",
"frozen_encoder.backbone.model.blocks.11.norm1.weight",
"frozen_encoder.backbone.model.blocks.11.norm1.bias",
"frozen_encoder.backbone.model.blocks.11.attn.qkv.weight",
"frozen_encoder.backbone.model.blocks.11.attn.qkv.bias",
"frozen_encoder.backbone.model.blocks.11.attn.proj.weight",
"frozen_encoder.backbone.model.blocks.11.attn.proj.bias",
"frozen_encoder.backbone.model.blocks.11.ls1.gamma",
"frozen_encoder.backbone.model.blocks.11.norm2.weight",
"frozen_encoder.backbone.model.blocks.11.norm2.bias",
"frozen_encoder.backbone.model.blocks.11.mlp.fc1.weight",
"frozen_encoder.backbone.model.blocks.11.mlp.fc1.bias",
"frozen_encoder.backbone.model.blocks.11.mlp.fc2.weight",
"frozen_encoder.backbone.model.blocks.11.mlp.fc2.bias",
"frozen_encoder.backbone.model.blocks.11.ls2.gamma",
"frozen_encoder.backbone.model.norm.weight",
"frozen_encoder.backbone.model.norm.bias"]
    
hold_out_train_scenes = ['0004', '0007', '0035', '0235', '0036', '0061', '0057', '0122', '0341', '0189']
def segmented_normalize(mat, threshold=1000, ratio=0.8):
    mat = mat.to(torch.float32)
    result = torch.zeros_like(mat)

    max_val = torch.max(mat)
    if max_val <= threshold:
        # all normalized to 0-1
        return mat / max_val if max_val > 0 else mat

    # 0 ~ threshold to 0 ~ ratio
    mask_low = mat <= threshold
    result[mask_low] = (mat[mask_low] / threshold) * ratio

    # > threshold to ratio ~ 1
    mask_high = mat > threshold
    result[mask_high] = ((mat[mask_high] - threshold) / (max_val - threshold)) * (1 - ratio) + ratio

    return result


def regularization(model, model_old):
    loss_reg = 0
    params_new = dict(model.named_parameters())
    params_old = dict(model_old.named_parameters())

    for name, p_new in params_new.items():
        if not name.startswith('aggregator'): continue
        if name in params_old and p_new.shape == params_old[name].shape:
            loss_reg += torch.sum((p_new - params_old[name]) ** 2)
    return loss_reg


def symmetric_inf(matrix, threshold=0., reverse=False):
    i_upper = np.triu_indices_from(matrix, k=1)
    if threshold > 0: matrix[i_upper[0], i_upper[1]] = segmented_normalize_np(matrix[i_upper], threshold)
    matrix[i_upper[1], i_upper[0]] = matrix[i_upper]
    if reverse: 
        matrix = 1 - matrix
        np.fill_diagonal(matrix, np.inf)
    else:
        np.fill_diagonal(matrix, -np.inf)

    return matrix

def segmented_normalize_np(mat, threshold=1000, ratio=0.8):
    mat = mat.astype(np.float32)
    result = np.zeros_like(mat)

    max_val = np.max(mat)
    if max_val <= threshold:
        # all normalized to 0-1
        return mat / max_val if max_val > 0 else mat

    # 0 ~ threshold to 0 ~ ratio
    mask_low = mat <= threshold
    result[mask_low] = (mat[mask_low] / threshold) * ratio

    # > threshold to ratio ~ 1
    mask_high = mat > threshold
    result[mask_high] = ((mat[mask_high] - threshold) / (max_val - threshold)) * (1 - ratio) + ratio

    return result


def split_file_name(name):
    files = re.findall(r"\w+\.(?:jpg|jpeg|png|gif)", name, re.IGNORECASE)
    return [f.lstrip("_") for f in files]


def load_config(config_path):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def save_config(config, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / "config.yaml", "w") as f:
        yaml.dump(config, f)


