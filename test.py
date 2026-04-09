import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataloaders.TestDataset import build_test_dataset

from src.kruskals import *
from src.test_utils import *
from src.train_util import *
from src.utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--class_model", default="best.ckpt")
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--src", default=None)
    parser.add_argument("--mode", default="test")
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--max_size", default=500, type=int)
    parser.add_argument("--cluster", action="store_true", default=False)
    parser.add_argument("--save_top", action="store_true", default=False)
    parser.add_argument("--save_knn", action="store_true", default=False)
    parser.add_argument("--ks", type=int, nargs="+", default=None)
    parser.add_argument("--dataset", default="megadepth")
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--extra", default="")
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--filter_nodes", action="store_true", default=False)
    parser.add_argument("--update_thr", default=0.9, type=float)
    parser.add_argument("--knn", default=5, type=int)

    return parser.parse_args()


args = parse_args()

if args.src is None:
    args.src = load_config("data_dirs.yaml")[args.dataset]

args.src = Path(args.src)

print(f"------------------start testing {args.class_model}---------------------")

scenes = os.listdir(args.src) if args.scenes is None else [args.scenes]
class_model = load_class_model(args.class_model, args.config_file)
pre = "" if args.dataset == "megadepth" else "images/"

out_dir = build_output_dir(args)

for scene in scenes:
    scene_dir = args.src / scene

    # Skip invalid scenes
    if (not scene_dir.is_dir()) or (not (scene_dir / "sfm").is_dir()):
        continue

    test_dataset, image_list = build_test_dataset(args, scene, pre=pre)
    test_data_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=200,
        shuffle=False,
        pin_memory=True,
    )

    N = len(image_list)

    # Compute all image tokens
    with torch.no_grad():
        tokens = []
        for image in test_data_loader:
            tokens.append(class_model.backbone(image.to(class_model.device)))
        tokens = torch.cat(tokens, dim=0)

    # edge score prediction
    predicted = infer_edges(tokens, class_model, args, out_dir, scene)

    probs = predicted["probs"]
    filtered_indicess = predicted.get("filtered_indices", None)

    if filtered_indicess is None:
        scores_matrix = probs
    else:
        # Convert edge predictions to a full score matrix
        scores_matrix = torch.zeros((N, N), device=probs.device, dtype=probs.dtype)
        scores_matrix[filtered_indicess[0], filtered_indicess[1]] = probs
        scores_matrix[filtered_indicess[1], filtered_indicess[0]] = probs

    ks = args.ks if args.ks is not None else [1, 2]

    # Optional: find unreliable nodes
    if args.filter_nodes:
        filtered_images, mask = find_unreliable_nodes(scores_matrix)
        scores_matrix = scores_matrix[mask][:, mask]
        image_list = [img for i, img in enumerate(image_list) if mask[i].item()]
        save_filtered_image_list(out_dir, scene, image_list)

    if args.save_top:
        selected_edges = save_mst_pairs_update_dists_numpy(
            scores_matrix,
            ks=ks,
            save_path=out_dir,
            image_list=image_list,
            suffix=scene,
            knn=args.knn,
            update_thr=args.update_thr
        )

    if args.save_knn:
        save_knn_pairs(
            scores_matrix,
            ks=args.ks,
            save_path=args.out_dir,
            image_list=image_list,
            remove_prefix=str(args.src),
            suffix=scene
        )
