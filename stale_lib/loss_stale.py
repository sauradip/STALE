# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml
from torch.autograd import Variable
from torch.nn.functional import normalize

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)


ce = nn.CrossEntropyLoss()

bce_cls = nn.BCEWithLogitsLoss()

em_loss = nn.CosineEmbeddingLoss(margin=0.2,reduction='sum')
cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
l2_dist = nn.PairwiseDistance(p=2)

lambda_1 = config['loss']['lambda_1']
lambda_2 = config['loss']['lambda_2']
split = config['dataset']['split']
nshot = config['fewshot']['shot']

if split == 50:
    alpha = 0.1
    beta = 1
    gamma = 1
    delta = 0.1
    sigma = 0.1
else:
    alpha = 0.4
    beta = 0.3
    gamma = 1
    delta = 0.1
    sigma = 1
    # alpha = 0.1
    # beta = 1
    # gamma = 1
    # delta = 0.1
    # sigma = 0.1

def top_lr_loss(target,pred, mode):

    gt_action = target
    pred_action = pred
    topratio = 0.6
    num_classes = 200
    alpha = 10
    fsmode = config['fewshot']['mode']
    nshot = config['fewshot']['shot']
    nway = config['fewshot']['num_way']

    if mode == "train":
        num_classes = 150
    else:
        num_classes = 50

    pmask = (gt_action == 1).float()
    nmask = (gt_action == 0).float()
    nmask = nmask 
    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    eps = 0.000001
    pred_p = torch.log(pred_action + eps)
    pred_n = torch.log(1.0 - pred_action + eps)

    topk = int(num_classes * topratio)
    count_pos = num_positive
    hard_neg_loss = -1.0 * (1.0-gt_action) * pred_n
    topk_neg_loss = -1.0 * hard_neg_loss.topk(topk, dim=1)[0] #topk_neg_loss with shape batchsize*topk
    loss = (gt_action * pred_p).sum() / count_pos + alpha*(topk_neg_loss.cuda()).mean()

    return -1*loss





class BinaryDiceLoss(nn.Module):
   
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

dice = BinaryDiceLoss()

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss


kl_loss = KLLoss()

def top_ce_loss(gt_cls, pred_cls):

    ce_loss = F.cross_entropy(pred_cls,gt_cls)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) **2 * ce_loss).mean()
    loss = focal_loss 

    # print("toploss2",loss)

    return loss




def bottom_branch_loss(gt_action, pred_action):

    pmask = (gt_action == 1).float()
    nmask = (gt_action == 0).float()
    nmask = nmask 
    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_action + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_action + epsilon) * nmask
    w_bce_loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    BCE_loss = F.binary_cross_entropy(pred_action,gt_action,reduce=False)
    pt = torch.exp(-BCE_loss)
    # F_loss = 0.4*loss2 + 0.6*dice(pred_action,gt_action)
    F_loss = lambda_2*w_bce_loss + (1 - lambda_2)*dice(pred_action,gt_action)
    
    return F_loss

def bg_embed_loss(embedding):

    B, C, T = embedding.shape
    bg_embed = embedding[:,C-1,:]
    # print(bg_embed)
    tar = Variable(torch.Tensor(embedding.size(0)).cuda().fill_(-1.0))
    loss_em = 0
    for i in range(0,C):
        cls_embed = embedding[:,i,:]
        loss_em += (cos_sim(cls_embed,bg_embed).mean())**2
    
    fin_loss = (loss_em / (C-1))



    return fin_loss

def Motion_MSEloss(output,clip_label,motion_mask=torch.ones(100).cuda()):
    z = torch.pow((output-clip_label),2)
    loss = torch.mean(motion_mask*z)
    return loss

def redundancy_loss(gt_action , pred_action, gt_cls, pred_cls, features):
    ### inter-branch consistency loss ## 
    mask_fg = torch.ones_like(gt_cls).cuda()
    mask_bg = torch.zeros_like(gt_cls).cuda()
    sim_loss = 0
    B,K,T = pred_cls.size()
    if T == 100 :
        for i in range(B):
            val_top,_ = torch.max(torch.softmax(pred_cls[i,:200,:],dim=0),dim=0)
            val_bot ,_ = torch.max(pred_action[i,:,:], dim=1)
            cls_thres = float(torch.mean(val_top,dim=0).detach().cpu().numpy())
            mask_thres = float(torch.mean(val_bot,dim=0).detach().cpu().numpy())
            top_mask = torch.where(val_top >= cls_thres, mask_fg[i,:], mask_bg[i,:]).cuda(0)
            bot_mask = torch.where(val_bot >= mask_thres, mask_fg[i,:], mask_bg[i,:]).cuda(0)
            top_loc = (top_mask==1).nonzero().squeeze().cuda(0)
            bot_loc = (bot_mask==1).nonzero().squeeze().cuda(0)
            top_feat = torch.mean(features[i,:,top_loc],dim=1).cuda(0)
            bot_feat = torch.mean(features[i,:,bot_loc],dim=1).cuda(0)

            sim_loss += (1-cos_sim(top_feat,bot_feat))
    const_loss = sim_loss / B


    fin_loss =  const_loss
   
    return fin_loss

def top_branch_loss(gt_cls, pred_cls, mask_gt, cls_pred, cls_gt, features, mode):

    cls_gt = cls_gt.type(torch.LongTensor).cuda()
    cls_gt_max = torch.argmax(cls_gt,dim=1)
    loss = 0.9*top_ce_loss(gt_cls.cuda(), pred_cls) + top_ce_loss(cls_gt_max, cls_pred) 

    return loss

def stale_loss(gt_cls, pred_cls ,gt_action , pred_action, mask_gt, bot_pred, bot_gt, cls_pred, cls_gt, features, mode):

    top_loss = top_branch_loss(gt_cls, pred_cls, mask_gt, cls_pred,cls_gt,features,mode)
    bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action) 
    red_loss = redundancy_loss(gt_action , pred_action, gt_cls, pred_cls, features)
    fg_loss =  F.binary_cross_entropy(bot_pred,bot_gt.cuda(),reduction='mean') 
    bg_loss = 0
    meta_class = config['fewshot']['meta_class']
    if not meta_class : ## learn both class and mask branch
        tot_loss = alpha*top_loss + beta*bottom_loss + gamma*bg_loss + delta*fg_loss + sigma*red_loss
    else: ## learn only class branch
        tot_loss = alpha*top_loss + beta*bottom_loss + gamma*bg_loss + delta*fg_loss + sigma*red_loss
        # tot_loss = 0.4*top_loss + 0.3*bottom_loss + bg_loss + 0.1*fg_loss + red_loss ## ---> gives SOTA in 0.75 and 2% above SOTA in 0.5 
        # tot_loss = 0.1*top_loss + bottom_loss + bg_loss + fg_loss + 0.1*red_loss ## ---> gives SOTA in 0.75 and 2% above SOTA in 0.5 
        # tot_loss = 0.1*top_loss + bottom_loss + bg_loss + fg_loss + 0.1*red_loss ### --> gives sota in 0.5 


    return tot_loss, top_loss, bottom_loss , fg_loss
