import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import pandas as pd
from src.utils import *
from models import helper
from PIL import Image
import numpy as np
from tqdm import tqdm
import h5py

from torch.nn.functional import cosine_similarity

class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = get_loss(loss_name)
        self.miner = get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
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
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)

            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        images, labels = batch # scene labels

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = images.shape
        
        # reshape places and labels
        images = images.view(BS*N, ch, h, w) # 240, 3, 224, 224
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')
        import pdb; pdb.set_trace()
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []



class PatchVPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', \
        loss_name_patch='MultiSimilarityLoss',
        miner_margin=0.1,
        faiss_gpu=False,
        alpha =0.5, 
        beta=0.5
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = get_loss(loss_name)
        self.patch_loss_fn = get_loss(loss_name_patch)
        self.miner = get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
        try:
            self.labels_saved = h5py.File("/mnt/personal/weitong/cache/GSVCities/backup/saved_roma_supervision_gsv_train.h5", 'r')
            self.patch_pair_list = np.loadtxt("data_preprocess/train_pairs_short_new.txt", usecols=[-2, -1], dtype=np.int16).reshape(-1, 60, 2)
        except:
            pass
        self.alpha = alpha
        self.beta = beta

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x, patch_x = self.aggregator(x)
        return x, patch_x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
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
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    def patch_loss(self, patch_descriptors, pair_indices, patch_labels):    
        # B*4, 128, 256; 60, 2; 60 list
        indices_tuple = []
        for i, index in enumerate(pair_indices):
            score_matrix = torch.ones((256, 256))
            score_matrix[patch_labels[i][:, 0], patch_labels[i][:, 1]] = 0
            score_matrix = torch.triu(score_matrix, diagonal=0)
            neg_label = score_matrix.nonzero()
            neg_label = neg_label[torch.randperm(neg_label.shape[0])[:patch_labels[i].shape[0]]]
            # distance = 1 - cosine_similarity(patch_descriptors[index[0]].T.unsqueeze(0), patch_descriptors[index[1]].T.unsqueeze(1), axis=-1)
            # pos_distances += [distance[patch_labels[i][:, 0], patch_labels[i][:, 1]]]    
            # neg_distances += [distance[neg_label[:, 0], neg_label[:, 1]]]  
            indices_tuple += [torch.concat((torch.from_numpy(patch_labels[i] * (i+1)), neg_label * (i+1)), dim=-1)]  
        import pdb; pdb.set_trace()
        return self.patch_loss_fn(patch_descriptors[pair_indices[:, 0]].transpose(-1, -2).flatten(0, 1), 
                                  indices_tuple=tuple(i for i in torch.vstack(indices_tuple).T), 
                                  ref_emb=patch_descriptors[pair_indices[:, 1]].transpose(-1, -2).flatten(0, 1))
            

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            import pdb; pdb.set_trace()
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            # pos_pairs = torch.vstack((miner_outputs[0], miner_outputs[1])).T
            # neg_pairs = torch.vstack((miner_outputs[2], miner_outputs[3])).T

            # sorted_neg_pairs = torch.sort(neg_pairs, dim=1).values
            # unique_neg_pairs = torch.unique(sorted_neg_pairs, dim=0)

            # sorted_pos_pairs = torch.sort(pos_pairs, dim=1).values
            # unique_pos_pairs = torch.unique(sorted_pos_pairs, dim=0)
            # min_num_pairs = min(unique_pos_pairs.shape[0], unique_neg_pairs.shape[0])
            # if min_num_pairs >30:
            #     min_num_pairs = 30
            # torch.manual_seed(42)
            # random_pos_indices = torch.randperm(unique_pos_pairs.size(0), device=unique_pos_pairs.device)
            # random_indices = torch.randperm(unique_neg_pairs.size(0), device=unique_neg_pairs.device)
            # pairs = torch.concat((unique_pos_pairs[random_pos_indices[:min_num_pairs]], unique_neg_pairs[random_indices[:min_num_pairs]]), dim=0)

            # with open('train_pairs_short_new.txt', 'a') as file:
            #     for p in pairs:
            #         file.write(f"{paths_all[p[0]]} {paths_all[p[1]]} {str(p[0].item())} {str(p[1].item())}")
            #         file.write("\n")
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value

            # TODO: load roma labels, compute contrastive loss on patch matches
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):

        places, labels, image_paths = batch # scene labels
        paths_all = [pth for batch_path in image_paths for pth in batch_path]

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        # masks = [pair[0] in paths_all and pair[1] in paths_all for pair in self.train_patch_pairs[batch_idx]]
        # pair_indices = np.vstack([(paths_all.index(pair[0]), paths_all.index(pair[1])) for pair in self.train_patch_pairs[batch_idx][masks]])        # reshape places and labels
        # print(np.sum(masks))
            # reshape places and labels
        images = places.view(BS*N, ch, h, w) # 240, 3, 224, 224
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors, patch_descriptors = self(images) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')
        
        # return also the positive and negative pair indices
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        pair_indices = self.patch_pair_list[batch_idx]
        # import pdb; pdb.set_trace()
        # for pair in pair_indices:
        #     if paths_all[pair[0]].replace('/', '_') +'_' + paths_all[pair[1]].replace('/', '_') not in self.labels_saved.keys():
        #         print("not found",  pair, paths_all[pair[0]].replace('/', '_') +'_' + paths_all[pair[1]].replace('/', '_') )
        # import pdb; pdb.set_trace()

        patch_labels = [self.labels_saved[paths_all[pair[0]].replace('/', '_') +'_' + paths_all[pair[1]].replace('/', '_')]['roma_indices'].__array__() for pair in pair_indices]
        patch_loss = self.patch_loss(patch_descriptors, pair_indices, patch_labels)
        self.log('main_loss', loss.item(), logger=True, prog_bar=True)
        final_loss = self.alpha*loss + self.beta*patch_loss
        self.log('loss', final_loss.item(), logger=True, prog_bar=True)
        self.log('patch_loss', patch_loss.item(), logger=True, prog_bar=True)
        return {'loss': final_loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors, _ = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []

    def patch_label_generator(self, pair_paths, paths):
        gt_labels = []
        for (im1_idx, im2_idx) in tqdm(pair_paths):

            im1_path = paths[im1_idx.item()]
            im2_path = paths[im2_idx.item()]

            W_A, H_A = Image.open(im1_path).size
            W_B, H_B = Image.open(im2_path).size

            warp, certainty = self.roma_model.match(im1_path, im2_path, device='cuda')
            matches, certainty = self.roma_model.sample(warp, certainty)
            kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)  

            scale0 = torch.tensor([224./W_A, 224./H_A]).to(kpts1.device)
            scale1 = torch.tensor([224./W_B, 224./H_B]).to(kpts1.device)

            patches_roma_pts1 = self.patch_helper.find_patch(kpts1*scale0)
            patches_roma_pts2 = self.patch_helper.find_patch(kpts2*scale1)

            match_patches = self.patch_helper.match_patch(patches_roma_pts1, patches_roma_pts2,all=False, N=1)
            match_patch_ids = torch.tensor([m[0] for m in match_patches])
            match_patch_conf = torch.tensor([m[1] for m in match_patches])
            unique_idx = torch.unique(match_patch_ids[:, 0], return_inverse=False)# indices where it is unique
            unique_idx_back = torch.unique(match_patch_ids[:, 1], return_inverse=False)

            plot_indices = []
            plot_indices_back = []

            for u_idx in unique_idx:
                many_list = (match_patch_ids[:, 0] == u_idx).nonzero(as_tuple=True)[0]
                chosen_idx = match_patch_ids[many_list][torch.argmax(match_patch_conf[many_list])]
                plot_indices += [chosen_idx]

            for u_idx in unique_idx_back:
                many_list = (match_patch_ids[:, 1] == u_idx ).nonzero(as_tuple=True)[0] # indices to the same query patch
                chosen_idx = match_patch_ids[many_list[torch.argmax(match_patch_conf[many_list])]]
                plot_indices_back += [chosen_idx]

            final = []
            for final_idx in plot_indices:
                if any(final_idx[0] == back_indices[0] and final_idx[1] == back_indices[1] for back_indices in plot_indices_back):
                    final += [final_idx]
            gt_labels += [torch.vstack(final)]

        return gt_labels
    

# TODO: add captioning+Blip module


class TextVPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = get_loss(loss_name)
        self.miner = get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
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
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)

            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch # scene labels
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w) # 240, 3, 224, 224
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')

        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []



class ClassVPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = get_loss(loss_name)
        self.miner = get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
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
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch # scene labels
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w) # 240, 3, 224, 224
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')

        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []


class ScoreModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = get_loss(loss_name)
        self.miner = get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
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
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)

            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        images, scene, image_paths = batch # scene labels

        scene_labels = np.loadtxt(f"/mnt/personal/weitong/cache/megadepth_runs/2024-12-07_23-23-37/{scene[0]}/{scene[0]}/inliers.txt", dtype=object)
        scene_labels = pd.DataFrame(scene_labels, columns=['File1', 'File2', 'Value'])
        image_paths =  [p[0] for p in image_paths]
        BS, N, ch, h, w = images.shape
        
        # Feed forward the batch to the model
        descriptors, edge_indices, predicted_edges = self(images.view(N, ch, h, w)) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')
        
        gt_a_matrix = torch.zeros(N, N)
        for i, query in enumerate(image_paths):
            for j, dst in enumerate(image_paths):
                if query == dst:
                    pass
                else:
                    try:
                        if ((scene_labels['File1'] == query) & (scene_labels['File2'] == dst)).sum() == 1:
                            gt_a_matrix[i, j] = float(scene_labels[(scene_labels['File1'] == query) & (scene_labels['File2'] == dst)]['Value'].values[0])
                        else:
                            gt_a_matrix[j, i] = float(scene_labels[(scene_labels['File1'] == dst) & (scene_labels['File2'] == query)]['Value'].values[0])
                    except:
                        pass
                        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        positive_mask = gt_a_matrix.nonzero()
        negative_mask = (gt_a_matrix ==0).nonzero()

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')
        num = min(len(positive_mask), len(negative_mask))

        # TODO: add predicted_edges to loss
        indices_tuple = (positive_mask[:num, 0], positive_mask[:num, 1], negative_mask[:num, 0], negative_mask[:num, 1])

        loss = self.loss_fn(descriptors, indices_tuple=indices_tuple)
        
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []

class PosModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = get_loss(loss_name)
        self.miner = get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
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
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)

            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        images, labels = batch # scene labels

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = images.shape
        
        # reshape places and labels
        images = images.view(BS*N, ch, h, w) # 240, 3, 224, 224
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above
        positives = (labels > 0).nonzero().flatten()
        negatives = (labels <= 0).nonzero().flatten()

        positive_mask = torch.from_numpy(np.vstack([(2*p.item(), 2*p.item()+1) for p in positives]))
        negative_mask = torch.from_numpy(np.vstack([(2*p.item(), 2*p.item()+1) for p in negatives]))

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')
        num = min(len(positive_mask), len(negative_mask))
        indices_tuple = (positive_mask[:num, 0], positive_mask[:num, 1], negative_mask[:num, 0], negative_mask[:num, 1])

        loss = self.loss_fn(descriptors, indices_tuple=indices_tuple)
        
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []