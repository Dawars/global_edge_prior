import pytorch_lightning as pl
from models.class_model_new import EdgeRankModel
from dataloaders.MegadepthDataloader import MegadepthClassDataModule
from pytorch_lightning.callbacks import Callback
import torch
from pytorch_lightning.loggers import CSVLogger
from src.train_util import load_config

class ClearCacheCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            print(f"Epoch {trainer.current_epoch}: CUDA cache cleared")
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()


if __name__ == '__main__':   
    pl.seed_everything(0, workers=True)    
    paths = load_config("data_dirs.yaml") 
    datamodule = MegadepthClassDataModule(
        src=paths['megadepth_src'],
        gt_src_path=paths['megadepth_gt_src_path'],
        batch_size=1,
        shuffle_all=True, 
        image_size=(224, 224),
        num_workers=5,
        val_set_names=['megadepth'],
        num_image_per_scene=240,
        resample=True, 
    )
    
    model = EdgeRankModel(
        edge_arch='Custom_GNN',
        edge_config={
            'dropout_mlp':0.5,
            'num_classes':1,
            'return_edges':False,
            'knn':-1
        },
        lr = 1e-5,
        optimizer='adamw',
        weight_decay=1e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 6000,
        },
        loss_name='NDCG',
        loss_config = {'k':3000, 'reduction': 'mean', 'mu':50, 'sigma':5},
        threshold=20,
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filename='best',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=True,
        mode='min'  
    )

    clear_cache_cb = ClearCacheCallback()
    csv_logger = CSVLogger(
        save_dir="trained_models",
        name=model.loss_name, 
        flush_logs_every_n_steps=50,
        version=None 
    )

    trainer = pl.Trainer(
        # accumulate_grad_batches=4,
        accelerator='gpu',
        devices=1,
        default_root_dir=f'trained_models/{model.loss_name}',
        num_nodes=1,
        num_sanity_val_steps=0,
        precision='32', 
        max_epochs=50,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb, clear_cache_cb],
        reload_dataloaders_every_n_epochs=0, 
        log_every_n_steps= 10,
        logger=csv_logger

    )

    trainer.fit(model=model, datamodule=datamodule)
