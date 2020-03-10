import torch
import torch.nn as nn
import numpy as np

class BaS_Net_loss(nn.Module):
    def __init__(self, alpha):
        super(BaS_Net_loss, self).__init__()
        self.alpha = alpha  #正则化项系数 0.0001
        self.ce_criterion = nn.BCELoss()

    def forward(self, score_base, score_supp, fore_weights, label):
        loss = {}

        """
        label_base:
            [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1]]
        label_supp:
            [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]]
        """
        label_base = torch.cat((label, torch.ones((label.shape[0], 1)).cuda()), dim=1)  # 在最后添加背景类，base网络学习将背景类也区分出来
        # 因为抑制网络的输入移除了背景帧，因此抑制网络需要尽可能区分最后一个类别不是背景类
        label_supp = torch.cat((label, torch.zeros((label.shape[0], 1)).cuda()), dim=1) 
        
        """
        label_base:
            [[0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0.5],[0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0.5]]
        label_supp:
            [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]]
        """
        label_base = label_base / torch.sum(label_base, dim=1, keepdim=True)  #平均每个类别的权重
        label_supp = label_supp / torch.sum(label_supp, dim=1, keepdim=True)

        loss_base = self.ce_criterion(score_base, label_base)  # score_base : (B,21) label_base: (B,21)
        loss_supp = self.ce_criterion(score_supp, label_supp)
        loss_norm = torch.mean(torch.norm(fore_weights, p=1, dim=1))  #返回输入张量给定维dim 上每行的p 范数。 

        loss_total = loss_base + loss_supp + self.alpha * loss_norm

        loss["loss_base"] = loss_base
        loss["loss_supp"] = loss_supp
        loss["loss_norm"] = loss_norm
        loss["loss_total"] = loss_total

        return loss_total, loss

def train(net, train_loader, loader_iter, optimizer, criterion, logger, step):
    net.train()
    try:
        _data, _label, _, _, _ = next(loader_iter)
    except:
        loader_iter = iter(train_loader)
        _data, _label, _, _, _ = next(loader_iter)

    _data = _data.cuda()  # (B,750,2048)
    _label = _label.cuda()  # (B,anchor_num,20)

    optimizer.zero_grad()

    # score_base为未经抑制获得的视频级类别预测结果， score_supp为经过背景帧抑制后获得的视频分类结果
    score_base, _, score_supp, _, fore_weights = net(_data)  # (B, C+1), (B, C+1)

    cost, loss = criterion(score_base, score_supp, fore_weights, _label)  # cost：total_loss   loss:{'loss_base','loss_supp','loss_norm'}

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)
