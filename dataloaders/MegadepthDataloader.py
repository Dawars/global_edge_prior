import pytorch_lightning as pl
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
from dataloaders.MegadepthDataset import MegadepthDataset


IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]}


class MegadepthClassDataModule(pl.LightningDataModule):
    def __init__(self,
                 src=None,
                 gt_src_path=None,
                 batch_size=32,
                 shuffle_all=False,
                 image_size=(480, 640),
                 num_workers=4,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 val_set_names=['pitts30k_val', 'msls_val'],
                 num_image_per_scene=-1,
                 resample=False,

                 ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.val_set_names = val_set_names
        self.num_image_per_scene = num_image_per_scene
        self.resample = resample
        self.gt_src_path = gt_src_path
        self.src = src
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightning
        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all
            }

        self.valid_loader_config = {
            'batch_size': 1,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader with reload routine
            self.reload()
            # load validation sets
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if valid_set_name.lower() == 'megadepth':
                    self.val_datasets.append(MegadepthDataset(
                        gt_src_path=self.gt_src_path,
                        src=self.src,
                        transform=self.train_transform,
                        num_image_per_scene=self.num_image_per_scene,
                        val=True,
                        resample=self.resample
                        ))


    def reload(self):
        self.train_dataset = MegadepthDataset(
            gt_src_path=self.gt_src_path,
            src=self.src,
            transform=self.train_transform,
            num_image_per_scene=self.num_image_per_scene,
            resample=self.resample)

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders
