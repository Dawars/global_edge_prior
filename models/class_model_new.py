import torch
import itertools
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
from src.utils import *
from src.losses import *
from models import helper
from torchmetrics import AveragePrecision
import torch.nn.functional as F
import pytorch_lightning as pl
from src.train_util import segmented_normalize, regularization
from models.aggregators.megaloc import MegaLocModel


def global_tokens2scores(descriptors):
    scores = torch.einsum('id,jd->ij', descriptors, descriptors) 
    N = descriptors.shape[0]
    diagonal = torch.eye(N, N).bool().to(scores.device)
    scores.masked_fill_(diagonal, -torch.inf)
    return scores

class EdgeRankModel(pl.LightningModule):

    def __init__(self,
        #---- Aggregator
        agg_arch='MegsLoc',
        agg_config={},
        
        #---- Edge classifier
        edge_arch='MLP',
        edge_config={},
        #---- Train hyperparameters
        lr=1e-5, 
        optimizer='adamw',
        weight_decay=1e-4,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='NDCG', 
        loss_config={'k':3000, 'reduction': 'mean', 'mu':50, 'sigma':5},
        binary_labels=False, 
        threshold=10., 
        test=False, 
    ):
        super().__init__()
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Edge classifier
        self.edge_arch = edge_arch
        self.edge_config = edge_config
        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = get_loss(loss_name, loss_config)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.binary_labels = binary_labels
        self.threshold=threshold
        if test:
            self.backbone = MegaLocModel()
        else:
            self.backbone = torch.hub.load("gmberton/MegaLoc", "get_trained_model").cuda()
            self.backbone.train()
            self.average_precision = AveragePrecision(task="binary")

        for p in self.backbone.parameters():
            p.requires_grad = True
        for name, param in self.backbone.named_parameters():
            if name.startswith("backbone"):
                param.requires_grad = False
        self.edge_classifier = helper.get_classifier(edge_arch, edge_config)
        self.aggregate_weights=None

    # the forward pass of the lightning model
    def forward(self, x, aggre_weights=None):

        x = self.backbone(x)    
        x = self.edge_classifier(x, aggre_weights=aggre_weights)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        
        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    def initialize_agg_weights(self, images):
        with torch.no_grad():
            tokens = self.backbone(images)
            S = global_tokens2scores(tokens)
            top_k = self.edge_config['knn'] if len(S) > self.edge_config['knn'] else len(S)

            topk_indices = torch.topk(S, top_k).indices
            N = S.size(0)

            top_mask = torch.zeros_like(S, dtype=torch.bool)  # [N, N]
            rows = torch.arange(N, device=S.device).unsqueeze(1).repeat(1, top_k)  # [N, K]
            cols = topk_indices  # [N, K]
            top_mask[rows, cols] = True

            mask = torch.triu(torch.ones_like(S, dtype=torch.bool), diagonal=1)
        return (S *top_mask)[mask]
    
    def initialize_agg_weights_tokens(self, tokens):
        with torch.no_grad():
            S = global_tokens2scores(tokens)
            top_k = self.edge_config['knn'] if len(S) > self.edge_config['knn'] else len(S)
            topk_indices = torch.topk(S, top_k).indices
            N = S.size(0)

            top_mask = torch.zeros_like(S, dtype=torch.bool)  # [N, N]
            rows = torch.arange(N, device=S.device).unsqueeze(1).repeat(1, top_k)  # [N, K]
            cols = topk_indices  # [N, K]
            top_mask[rows, cols] = True

            mask = torch.triu(torch.ones_like(S, dtype=torch.bool), diagonal=1)
        return (S *top_mask)[mask]
    
    def training_step(self, batch):

        total_loss = 0
        BS = len(batch[0])   # batch size (# scenes)
        for b in range(BS):
            one_batch = [data[b].unsqueeze(0) for data in batch]
            out = self._process_one_scene(one_batch, b)
            total_loss += out[f'loss_{b}']

        self.log("loss", total_loss/BS)
        return {"loss": total_loss/BS}

    def _process_one_scene(self, batch, batch_id):
        images = batch[0]
        gt_labels = batch[1]
        gt_labels_class = batch[2]
        total_loss = 0
        BS, N, ch, h, w = images.shape
        if (self.edge_config['knn'] > 0): 
            self.aggregate_weights = self.initialize_agg_weights(images.view(BS*N, ch, h, w))
        # Feed forward the batch to the model
        results = self(images.view(BS*N, ch, h, w), aggre_weights=self.aggregate_weights)
        output, edge_indices = results['edges'], results['edge_indices']
        indices = list(range(BS*N))
        combinations = list(itertools.combinations(indices, 2))
        all_edge_indices = torch.tensor(combinations).T

        assert (all_edge_indices.to('cuda')==edge_indices).all()
        full_label = torch.zeros(BS*N, BS*N).to(gt_labels.device)

        for i in range(BS):
            start = i * N
            end = (i + 1) * N
            full_label[start:end, start:end] = gt_labels[i]

        gt_labels = full_label[edge_indices[0], edge_indices[1]]
        predicted_labels = torch.sigmoid(output.squeeze()).to(gt_labels.device)
        binary_gt_labels = (gt_labels_class[0][edge_indices[0], edge_indices[1]].to(gt_labels.device) >self.threshold) + 0.

        loss = self.loss_fn(predicted_labels, gt_labels) 
        ap_score = self.average_precision(predicted_labels, binary_gt_labels.to(torch.int32))
        total_loss += loss

        self.log(f'class_loss_{batch_id}', loss.item(), logger=True, prog_bar=True)
        self.log(f'loss_{batch_id}', total_loss, logger=True, prog_bar=True)
        self.log(f'ap-score_{batch_id}', ap_score.item(), logger=True, prog_bar=True)

        return {f'loss_{batch_id}': total_loss}
    
    def on_train_epoch_end(self):
        self.batch_acc = []
    
    
    def validation_step(self, batch):

        total_loss = 0
        BS = len(batch[0])  
        for b in range(BS):
            one_batch = [data[b].unsqueeze(0) for data in batch]
            out = self._process_one_scene_val(one_batch, b)
            total_loss += out[f"val_loss_{b}"]

        self.log("val_loss", total_loss/BS)
        return {"val_loss": total_loss/BS}
    
    def _process_one_scene_val(self, batch, batch_idx):

        images = batch[0]
        gt_labels = batch[1]
        gt_labels_class = batch[2]
        BS, N, ch, h, w = images.shape
        if self.edge_config['knn'] > 0: 
            self.aggregate_weights = self.initialize_agg_weights(images.view(BS*N, ch, h, w))
        # Feed forward the batch to the model
        with torch.no_grad():
            results = self(images.view(BS*N, ch, h, w), aggre_weights=self.aggregate_weights)
        output, edge_indices = results['edges'], results['edge_indices']
        indices = list(range(BS*N))
        combinations = list(itertools.combinations(indices, 2))
        all_edge_indices = torch.tensor(combinations).T
        assert (all_edge_indices.to('cuda')==edge_indices).all()

        full_label = torch.zeros(BS*N, BS*N).to(gt_labels.device)
        for i in range(BS):
            start = i * N
            end = (i + 1) * N
            full_label[start:end, start:end] = gt_labels[i]

        gt_labels = full_label[all_edge_indices[0], all_edge_indices[1]]
        predicted_labels = torch.sigmoid(output.squeeze()).to(gt_labels.device)
        binary_gt_labels =  (gt_labels_class[0][all_edge_indices[0], all_edge_indices[1]] > self.threshold) + 0.

        loss = self.loss_fn(predicted_labels, gt_labels) 
        ap_score = self.average_precision(predicted_labels, binary_gt_labels.to(torch.int32))
        total_loss = loss

        self.log(f'val_class_loss_{batch_idx}', loss.item(), logger=True, prog_bar=True)
        self.log(f'val_loss_{batch_idx}', total_loss, logger=True, prog_bar=True)
        self.log(f'val_ap-score_{batch_idx}', ap_score.item(), logger=True, prog_bar=True)

        return {f'val_loss_{batch_idx}': total_loss}

