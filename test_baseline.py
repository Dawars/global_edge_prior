import torch
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import h5py
import types
from src.test_utils import *
from src.kruskals import *
from src.test_utils import *
from models.backbones.dinov2 import DINOv2
from dataloaders.TestDataset import build_test_dataset
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

with open("./data_dirs.yaml", "r") as f:
    config = yaml.safe_load(f)

if not os.path.exists(config['hloc_path']):
    print("Warnings: No HLoc loaded, SP+LG features cannot be computed")
else:
    sys.path.append(config['hloc_path'])
    from hloc import extract_features

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", default=None)
    parser.add_argument("--dataset", default='megadepth')
    parser.add_argument("--scene", default=None, nargs='+')
    parser.add_argument("--name", default='netvlad', nargs='+')
    parser.add_argument("--ks", type=int, nargs='+', default=None)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--mode", default='test')
    parser.add_argument("--out_dir", default='logs/megaloc_pairs')
    parser.add_argument("--knn", action='store_true', default=False)

    args = parser.parse_args()
    return args

args = parse_args()
paths = types.SimpleNamespace()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pre = '' if args.dataset == 'megadepth' else 'images/'
args.src = Path(args.src)

for model_i, name in enumerate(args.name):
    print(f"start with method {args.name[model_i]}")
    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    for scene in args.scene:
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

        ori_images = []
        images = []
        N = len(image_list)
        if (args.name[model_i] == 'netvlad') or (args.name[model_i] == 'cosplace'):
            conf = extract_features.confs[args.name[model_i]]
            retrieval_features = Path(args.out_dir) / f'{scene}_features.h5'
            if not os.path.exists(retrieval_features):
                extract_features.main(conf, Path(args.src)/scene/pre, feature_path=retrieval_features, image_list=image_list)
            with h5py.File(str(retrieval_features), "r") as hfile:
                desc = [hfile[n]['global_descriptor'].__array__() for n in image_list]
            global_descriptors = torch.from_numpy(np.stack(desc, 0)).float()

        elif args.name[model_i] == 'dinov2':
            test_model = DINOv2(num_trainable_blocks=0, return_token=True).cuda()
            cls_tokens =[]
            with torch.no_grad():
                for image in tqdm(test_data_loader):
                    img = image.to('cuda')
                    cls_tokens += test_model(img)[-1].cpu()
                    del img
                torch.cuda.empty_cache()
            global_descriptors = torch.vstack(cls_tokens)

        elif args.name[model_i] == 'megaloc':
            test_model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").cuda()
            with torch.no_grad():
                for image in tqdm(test_data_loader):
                    img = image.to('cuda')
                    images += [test_model(img).cpu()]
                    del img
                torch.cuda.empty_cache()
            global_descriptors = torch.concat(images, dim=0)

        elif args.name[model_i] == 'salad':
            test_model = torch.hub.load("serizba/salad", "dinov2_salad").cuda()
            with torch.no_grad():
                for image in tqdm(test_data_loader):
                    img = image.to('cuda')
                    images += [test_model(img).cpu()]
                    del img
                torch.cuda.empty_cache()
            global_descriptors = torch.concat(images, dim=0)

        scores = global_tokens2scores(global_descriptors)
        assert (scores == scores.T).all()
        diagonal = torch.eye(N, N).bool().to(scores.device)
        scores.masked_fill_(diagonal, -torch.inf)
        np.savez(f"{args.out_dir}/{scene}_scores.npz", **{'scores': scores.cpu().numpy()})
        subset_name = args.image_list.split('/')[-1].split('.')[0]

        # save pairs selected by kNN or MSTs
        if args.knn:
            save_knn_pairs(
                scores,
                ks=args.ks,
                save_path=args.out_dir,
                image_list=image_list,
                remove_prefix=str(args.src),
                scene=scene
                )
        else:
            save_mst_pairs(
                scores.cpu().numpy(),
                ks=args.ks,
                save_path=args.out_dir,
                image_list=image_list,
                scene=scene,
                suffix=subset_name
                )
