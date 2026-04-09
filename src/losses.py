from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from pytorch_metric_learning.reducers import PerAnchorReducer
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLRANK_ROOT = str(ROOT / "external/allrank")
if ALLRANK_ROOT not in sys.path:
    sys.path.append(ALLRANK_ROOT)

from allrank.models.losses.lambdaLoss import lambdaLoss
def get_loss(loss_name, loss_config=None):

    if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
    if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80) #these are params for image retrieval
    if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1, reducer=PerAnchorReducer())
    if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1, distance=DotProductSimilarity())
    if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)
    if loss_name == 'NTXentLoss': return losses.NTXentLoss(temperature=0.07) #The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
    if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
                                                                            swap=False,
                                                                            smooth_loss=False,
                                                                            triplets_per_anchor="all",)
    if loss_name == 'BCELoss': return nn.BCEWithLogitsLoss()
    if loss_name == 'MSELoss': return nn.MSELoss()
    if loss_name == 'BCELossnologits': return nn.BCELoss()
    if loss_name == 'FocalLoss': return FocalLoss(gamma=loss_config['gamma'], alpha=loss_config['alpha']) if loss_config != None else FocalLoss()
    elif loss_name =='NDCG': return NDCG(k=loss_config['k'], reduction=loss_config['reduction'], mu=loss_config['mu'], sigma=loss_config['sigma'])
    else:
        raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')

def get_miner(miner_name, margin=0.1):
    if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
    if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
    return None

class NDCG(nn.Module):
    def __init__(self, k=-1, reduction='mean', sigma=1., mu=10):
        super().__init__()
        self.k = k
        self.reduction = reduction
        self.sigma = sigma
        self.mu=mu
    def forward(self, pred_logits, labels):
        loss = lambdaLoss(pred_logits[None,:], labels[None, :], weighing_scheme="ndcgLoss2PP_scheme", reduction_log="binary", reduction=self.reduction, k=self.k, sigma=self.sigma, mu=self.mu)
        return loss

# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def get_torch_device():
    """Getter for an available pyTorch device.

    :return: CUDA-capable GPU if available, CPU otherwise
    """
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
