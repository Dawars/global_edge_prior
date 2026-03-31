import numpy as np
from models import aggregators
from models import backbones
from models import edge_predictors
import torch

def get_backbone(
        backbone_arch='resnet50',
        backbone_config={}
    ):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        backbone_config (dict, optional): this must contain all the arguments needed to instantiate the backbone class. Defaults to {}.

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        return backbones.ResNet(backbone_arch, **backbone_config)

    elif 'dinov2' in backbone_arch.lower():
        return backbones.DINOv2(model_name=backbone_arch, **backbone_config)

    elif 'dinov3' in backbone_arch.lower():
        return backbones.DINOv3(model_name=backbone_arch, **backbone_config)

def get_aggregator(agg_arch='ConvAP', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    
    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)
    
    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        assert 'in_h' in agg_config
        assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return aggregators.MixVPR(**agg_config)
        
    elif 'salad' in agg_arch.lower():
        assert 'num_channels' in agg_config
        assert 'num_clusters' in agg_config
        assert 'cluster_dim' in agg_config
        assert 'token_dim' in agg_config
        return aggregators.SALAD(**agg_config)
    

def get_classifier(
        edge_arch='dot',
        edge_config={}
    ):
    """Helper function that returns the backbone given its name

    Args:
        edge_arch (str, optional): . Defaults to 'dot'.
        edge_config (dict, optional): this must contain all the arguments needed to instantiate the edge classifier class. Defaults to {}.

    Returns:
        nn.Module: the backbone as a nn.Model object
    """

    if 'dot' in edge_arch.lower():
        return edge_predictors.WeightedDotProductClassifier(edge_arch, **edge_config) 
    elif 'gnn' in edge_arch.lower():
        return edge_predictors.custom_gnn.Custom_GNN_Predcitor(edge_arch, **edge_config)  


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    image_list_flat = [i for item in batch for i in item[-1]]
    merged_gt = {}
    for item in batch:
        merged_gt.update(item[1]) 

    return {
        "images": images,
        "image_list": image_list_flat,
        "GT": merged_gt,  
    }

def multi_classes_ranges(alpha, beta, K=5, smooth=True):
    """eqn(1) in https://openaccess.thecvf.com/content_cvpr_2018/papers/Fu_Deep_Ordinal_Regression_CVPR_2018_paper.pdf"""
    if smooth:
        epsilon = 1 - alpha
        alpha += epsilon
        beta += epsilon
    tis = [np.exp(np.log(alpha)+ np.log(beta/alpha)* i/(K)) for i in range(K+1)]

    return tis


def assign_labels(y, tis):
    """
    assign discrete labels for scalars

    """
    tis = np.array(tis)
    labels = np.searchsorted(tis, y, side='right') - 1
    labels = np.clip(labels, 0, len(tis) - 1)

    return labels