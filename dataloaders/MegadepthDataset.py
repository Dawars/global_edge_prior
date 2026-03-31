import os
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import h5py
import torch
import itertools
import numpy as np
from pathlib import Path
from src.train_util import *
from torch.utils.data import Dataset
import torchvision.transforms as T

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MegadepthDataset(Dataset):
    def __init__(self,
                transform=default_transform,
                num_image_per_scene=240,
                val=False,
                resample=True, 
                gt_src_path="/mnt/personal/weitong/cache/megadepth_runs_all/",
                src="/mnt/personal/weitong/olap_predictor_mlp_large/data/megadepth/"

        ):
        super(MegadepthDataset, self).__init__()
        
        self.src_path = Path(f"{gt_src_path}/val/") if val else Path(f"{gt_src_path}/train/")
        self.image_path_src = Path(f"{src}/reconstruct_images_val/") if val else Path(f"{src}/reconstruct_images/")
        self.transform = transform
        self.scenes = os.listdir(self.src_path)
        # self.scenes = [scene for scene in self.scenes if os.path.exists(self.src_path / scene / scene / "supervision.txt") and len(os.listdir(self.image_path_src / scene)) >= num_image_per_scene]
        for s in hold_out_train_scenes: 
            if s in self.scenes: self.scenes.remove(s)
        valid_scenes = []
        for scene in self.scenes:
            if os.path.exists(Path(gt_src_path)/f"gt_tri_inliers/{scene}_inliers.npz") and os.path.exists(Path(gt_src_path)/f"gt/{scene}_inliers.npz") and (len(os.listdir(self.image_path_src/ scene)) >= num_image_per_scene):
                valid_scenes.append(scene)
        self.scenes = valid_scenes

        self.image_lists = {}
        self.image_list_dict = {}
        self.gt_dict = {}
        self.gt_dict_class = {}
        self.train_scenes = []

        for scene in self.scenes:

            if num_image_per_scene!=-1:
                image_list = os.listdir(self.image_path_src /scene)[:num_image_per_scene]
                assert len(image_list) == num_image_per_scene
                self.image_lists[scene] = sorted(image_list)
            else:
                image_list = np.loadtxt(f"{gt_src_path}/image_list/{scene}.txt", dtype=object).tolist()
                self.image_lists[scene] = [img for img in image_list if os.path.exists(self.image_path_src/scene/img)]

            self.image_list_dict[scene] = np.loadtxt(f"{gt_src_path}/image_list/{scene}.txt", dtype=object).tolist()
            # normalized RANSAC inliers + triangulated inliers
            self.gt_dict[scene] = np.load(f"{gt_src_path}/gt/{scene}_inliers.npz")['inliers']
            # triangulated inliers, unnormalized
            self.gt_dict_class[scene] = np.load(f"{gt_src_path}/gt_tri_inliers/{scene}_inliers.npz")['inliers']

            if resample and len(os.listdir(self.image_path_src/ scene))>num_image_per_scene: 
                for i in range(len(os.listdir(self.image_path_src/ scene))//num_image_per_scene - 1):
                    if i > 16: break
                    self.image_lists[scene + f'_{i}'] = sorted(os.listdir(self.image_path_src / scene)[num_image_per_scene*(i+1):num_image_per_scene*(i+2)])
                    self.train_scenes += [scene + f'_{i}']

            self.train_scenes += [scene]

    def __getitem__(self, index):
        
        image_list = self.image_lists[self.train_scenes[index]]
        images = []
        image_path_src = self.image_path_src
        for image_path in image_list:
            img = self.image_loader(image_path_src/self.train_scenes[index].split('_')[0]/image_path)
            if self.transform is not None:
                img = self.transform(img)
            images += [img]

        scene = self.train_scenes[index].split('_')[0]
        image_list_save = self.image_list_dict[scene]

        # load the saved indices
        indices = {t: image_list_save.index(t) for t in image_list if t in image_list_save}
        picked_indices = [idx for idx in indices.values()]
        N = len(picked_indices)
        score_matrix = np.zeros((N, N), dtype=np.float32)
        score_matrix_class = np.zeros((N, N), dtype=np.float32)

        new_indices = list(range(N))
        new2old_indices = dict(zip(new_indices, picked_indices))

        # load the saved #inliers
        combinations = list(itertools.combinations(new_indices, 2))

        gt = self.gt_dict[scene]
        gt_class = self.gt_dict_class[scene]

        for i, j in combinations:
            score_matrix[i, j] = gt[new2old_indices[i], new2old_indices[j]]
            score_matrix_class[i, j] = gt_class[new2old_indices[i], new2old_indices[j]]

        return torch.stack(images), score_matrix, score_matrix_class
            
    def __len__(self):
        return len(self.train_scenes)


    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', self.transform.image_size)
        
def _to_path(p):
    return p if isinstance(p, Path) else Path(p)


class AnyData(Dataset):
    def __init__(self,
            src_path,
            src_val_path,
            image_path_src,
            image_path_val_src,
            gt_file_path, 
            gt_class_file_path, # number of triangulated inliers
            image_list_path, 
            val=False, 
            transform=default_transform,
        ):
        super(AnyData, self).__init__()

        src_path = _to_path(src_path)
        src_val_path = _to_path(src_val_path)
        image_path_src = _to_path(image_path_src)
        image_path_val_src = _to_path(image_path_val_src)
        gt_file_path = _to_path(gt_file_path)
        gt_class_file_path = _to_path(gt_class_file_path)
        image_list_path = _to_path(image_list_path)

        # paths
        self.src_path = src_path if not val else src_val_path
        self.image_path_src = image_path_src if not val else image_path_val_src
        self.val=val

        # load list of scenes
        self.scenes = [scene.replace('.txt', '') for scene in os.listdir(self.src_path)]

        self.image_lists = {}
        self.gt_dict = {}
        self.gt_dict_class = {}
        self.train_scenes = []
        
        for scene in self.scenes:
            scene_name = scene.split('_')[0]
            if not os.path.exists(gt_file_path/f"{scene_name}_inliers.npz"): continue
            score_matrix = np.load(gt_file_path/f"{scene_name}_inliers.npz")['inliers'] 
            class_score_matrix = np.load(gt_class_file_path/f"{scene_name}_inliers.npz")['inliers']

            all_image_list = np.loadtxt(image_list_path/f"{scene_name}.txt", dtype=str).tolist()
            subset_image_list = np.loadtxt(self.src_path/f"{scene}.txt", dtype=str).tolist()
            # convert the inlier matrix to a subset, accoring to the subset image list
            subset_idx = [all_image_list.index(img) for img in subset_image_list if img in all_image_list]

            if class_score_matrix.sum() != 0 and len(subset_image_list) < 300:   
                self.train_scenes += [scene]
                self.image_lists[scene] = subset_image_list
                self.gt_dict[scene] = score_matrix[np.ix_(subset_idx, subset_idx)]
                self.gt_dict_class[scene] = class_score_matrix[np.ix_(subset_idx, subset_idx)]
        self.transform = transform

    def __getitem__(self, index):

        image_list = self.image_lists[self.train_scenes[index]]
        images = []
        scene_name = self.train_scenes[index].split('_')[0]
        scene = self.train_scenes[index]
        for image_path in image_list:
            img = self.image_loader(self.image_path_src/scene_name/"images"/image_path) 
            if self.transform is not None:
                img = self.transform(img)
            images += [img]

        score_matrix = self.gt_dict[scene]

        return torch.stack(images), score_matrix, self.gt_dict_class[scene]

    def __len__(self):
        return len(self.train_scenes)


    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', self.transform.image_size)


