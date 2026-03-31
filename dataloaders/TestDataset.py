from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import os

def input_transform(image_size=None):
    MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    
def is_image_file(filename):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_extensions

def load_image_paths(src):
    image_list = []
    for root, dirs, files in os.walk(src):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_file(file_path): image_list += [file_path]
    image_list = sorted(image_list)
    return image_list
    
class ImageDataset(Dataset):
    def __init__(self, image_paths, image_names=None, image_size=(322, 322), check_valid=False):

        self.transform = input_transform(image_size=image_size)

        if image_names is None:
            image_names = [str(p) for p in image_paths]

        assert len(image_paths) == len(image_names)

        self.image_paths = [Path(p) for p in image_paths]
        self.image_list = list(image_names)  

        if check_valid:
            valid_paths = []
            valid_names = []
            for path, name in zip(self.image_paths, self.image_list):
                try:
                    with Image.open(path) as img:
                        img.verify()
                    valid_paths.append(path)
                    valid_names.append(name)
                except Exception:
                    print(f"Warning: Removing invalid image {path}")

            self.image_paths = valid_paths
            self.image_list = valid_names

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]).convert("RGB"))
        return image
    
def build_test_dataset(args, scene, pre='', image_size=(322, 322), check_valid=False):
    src = Path(args.src)

    # 1. image names & paths
    # if args.dataset not in ['megadepth', 'phototourism']:
    #     if args.image_list is not None:
    #         image_list = [x for x in np.loadtxt(args.image_list, dtype=object)]
    #         # remove prefix
    #         image_list = [img.replace(f"{scene}/", "") for img in image_list]
    #         cur_pre = '' if args.dataset == 'visym' else pre
    #         image_paths = [src.parent / scene / cur_pre / img for img in image_list]
    #     else:
    #         image_paths = [Path(p) for p in load_image_paths(src)]
    #         image_list = [str(p).replace(str(src), '').lstrip('/') for p in image_paths]

    # else:
    if args.image_list is not None:
        image_list = np.loadtxt(args.image_list, dtype=str).tolist()
        if isinstance(image_list, str):
            image_list = [image_list]
        image_paths = [src / scene / pre / img for img in image_list]

    else:
        image_paths = [Path(p) for p in load_image_paths(src / scene)]
        image_list = sorted([str(p).replace(str(src), '').replace(scene, '').replace(pre, '').lstrip('/') for p in image_paths])

    test_dataset = ImageDataset(
        image_paths=image_paths,
        image_names=image_list,
        image_size=image_size,
        check_valid=check_valid,
    )

    return test_dataset, test_dataset.image_list