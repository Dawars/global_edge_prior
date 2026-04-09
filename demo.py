
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from src.test_utils import *
from src.sfm_utils import compute_auc
from src.train_util import load_config
from dataloaders.TestDataset import ImageDataset
from pathlib import Path
config = load_config("data_dirs.yaml")
"""CUDA-12.4.0 PyTorch-Geometric Pillow/10.2.0 PyTorch-Lightning/2.5.2 torchvision/0.23.0 Open3D/0.18.0."""


# check if gpu device is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
scene = 'brandenburg_gate'

# load images
image_list = os.listdir('demo/images')
image_paths = [f'demo/images/{i}'for i in image_list]
print(f"--------------{len(image_list)} images loaded---------------------")

test_dataset = ImageDataset(image_paths=image_paths, image_names=image_list,)
test_data_loader = DataLoader(
        test_dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

# predict global edge scores
class_model = load_class_model('demo/best.ckpt', 'demo/best.yaml', device)

with torch.no_grad():
    tokens = []
    for image in test_data_loader:
        tokens.append(class_model.backbone(image.to(class_model.device)))
    tokens = torch.cat(tokens, dim=0) # N, 8448

N = len(tokens)
# edge score prediction
with torch.no_grad():
    predicted = class_model.edge_classifier(tokens)

edge_scores = torch.sigmoid(predicted['edges'].squeeze())
edge_indices = predicted['edge_indices']

# Convert edge predictions to a full score matrix
scores_matrix = torch.zeros((N, N), device=edge_scores.device, dtype=edge_scores.dtype)
scores_matrix[edge_indices[0], edge_indices[1]] = edge_scores
scores_matrix[edge_indices[1], edge_indices[0]] = edge_scores

os.makedirs('demo/pairs', exist_ok=True)
### save edges, for COLMAP
selected_edges = save_mst_pairs_update_dists_numpy(
            scores_matrix,
            ks=[1],
            save_path='demo/pairs',
            image_list=image_list,
        )

### TODO: run feature extraction, matching and geometric verification, then COLMAP mapping to finish the reconstruction.


### Optional: aggregate node scoers based on the edges
filtered_images, mask = find_unreliable_nodes(scores_matrix, topk=2)
scores_matrix = scores_matrix[mask][:, mask]
filtered_image_list = [img for i, img in enumerate(image_list) if mask[i].item()]
filtered_image_paths = [img for i, img in enumerate(image_paths) if mask[i].item()]

### remove outliers for VGGT
import sys
sys.path.append("external/vggt/")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()
model.requires_grad_(False)

# VGGT on filtered image set
images_tensor = load_and_preprocess_images(filtered_image_paths, mode="crop")
with torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else torch.no_grad():
    predictions = model(images_tensor.to(device))
save_predictions = {k: v.float().cpu() for k, v in predictions.items() if torch.is_tensor(v)}
np.savez('demo/predictions.npz', **save_predictions)

# VGGT on the whole image set
images_tensor_fil = load_and_preprocess_images(image_paths, mode="crop")
with torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else torch.no_grad():
    predictions = model(images_tensor_fil.to(device))
save_predictions = {k: v.float().cpu() for k, v in predictions.items() if torch.is_tensor(v)}
np.savez('demo/predictions_nofilter.npz', **save_predictions)

import pycolmap
def evaluate(vggt_result_path, image_list_used):
    save_predictions = np.load(vggt_result_path)
    image_hw = (518, 518)
    pose_enc = torch.from_numpy(save_predictions["pose_enc"])
    extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, image_hw)
    extrinsics = extrinsics.squeeze(0) if extrinsics.ndim == 4 else extrinsics
    Twc = convert_world_to_cam_to_cam_to_world(extrinsics)
    result = Twc.detach().cpu().float()

    Twc_pred = result.numpy()
    image_names = image_list
    sparse_gt_path = Path(f"{config['phototourism']}/{scene}/sfm/")
    sparse_gt = pycolmap.Reconstruction(sparse_gt_path)
    Twc_pred_dict = {
        image_list_used[i]: Twc_pred[i]
        for i in range(len(image_list_used))
    }
    # gt from sparse_gt
    Twc_gt_dict = {}
    for image_id, img in sparse_gt.images.items():
        Tcw = img.cam_from_world.matrix()
        Tcw4 = np.eye(4, dtype=np.float64)
        Tcw4[:3, :4] = Tcw
        Twc_gt_dict[img.name] = np.linalg.inv(Tcw4)

    # choose pairs
    valid_names = [name for name in image_list_used if name in Twc_gt_dict]

    # pairs = [(image_names[i], image_names[j]) for i in range(len(image_names)) for j in range(len(image_names)) if i != j]
    pairs = [
        (valid_names[i], valid_names[j])
        for i in range(len(valid_names))
        for j in range(len(valid_names))
        if i != j
    ]
    # evaluate
    dts, dRs, pairs = compute_rel_errors_from_poses(
        Twc_gt_dict=Twc_gt_dict,
        Twc_pred_dict=Twc_pred_dict,
        pairs=pairs
    )
    errors = [max(dt, dR) for dt, dR in zip(dts, dRs)]
    results = compute_auc(
            errors,
            [2.5, 5., 10., 20.],
            min_error=0.02,
        )
    return results

results_nofilter = evaluate("demo/predictions_nofilter.npz", filtered_image_list)
results_filter = evaluate("demo/predictions.npz", filtered_image_list)

import pandas as pd

df = pd.DataFrame(
    [
        ["VGGT", *results_nofilter],
        ["VGGT+Ours", *results_filter],
    ],
    columns=["Method", "AUC@2.5", "AUC@5", "AUC@10", "AUC@20"],
)
print(df.to_string(index=False))
