import torch
import os
from pathlib import Path
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from src.test_utils import *
import h5py 
import sys
from src.kruskals import *
import time
import types
sys.path.append('/mnt/personal/weitong/cache/Hierarchical-Localization/')
sys.path.append('/home/weitong/code/vop')

from hloc import extract_features
from src.test_utils import *
from models.backbones.dinov2 import DINOv2

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
        if scene == 'temple_nara_japan': continue
        
        if args.dataset not in ['megadepth', 'phototourism']:
            if args.image_list is not None: 
                image_list = [i for i in np.loadtxt(args.image_list, dtype=object)]
                image_list = [img.replace(f"{scene}/", '') for img in image_list]
                if (args.dataset == 'visym'): pre = ''
                test_dataset = MegaDataset(Path(str(args.src).replace(f"{scene}", '')), image_list, scene, pre=pre)
            else:
                image_list = load_image_paths(args.src)
                test_dataset = CustomDataset(image_list)
                image_list = test_dataset.image_list
                image_list = [img.replace(str(args.src), '' ).lstrip('/') for img in image_list]
        else:
            if args.image_list is not None:
                image_list = np.loadtxt(args.image_list, dtype=str).tolist()
            else:
                image_list = sorted([image for image in os.listdir(args.src / scene / pre)][:args.num_images]) if args.num_images!= -1 else sorted([image for image in os.listdir(args.src / scene/pre)])
            test_dataset = MegaDataset(args.src, image_list, scene, pre=pre)

        test_data_loader = DataLoader(test_dataset, num_workers=2, batch_size=200, shuffle=False, pin_memory=True)
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