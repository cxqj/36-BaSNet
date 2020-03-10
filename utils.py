import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import random
import config


def upgrade_resolution(arr, scale):  # (T,1,1), scale: 24
    x = np.arange(0, arr.shape[0])  
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')  # y = f(x)
    scale_x = np.arange(0, arr.shape[0], 1 / scale)  
    up_scale = f(scale_x)  # (18000,1,1)
    return up_scale

"""
tList: 得分大于指定阈值的position索引列表，预测的anchor就是索引列表中连续的区域
wtcam：拓展后每个position位置的类别得分(18000,1,1) 
final_score： 视频类别mask后的结果，例如：[0,0,0,1,0,0,0,0,0,1,0,0,0] 
c_pred: 当前视频预测动作类别索引,[3,9] 
scale:24 
v_len: 特征序列的长度 
sampling_frames ：25 
num_segments： 750
"""
def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, lambda_=0.25, gamma=0.2):
    # 24*750*25 当前拓展后每一个时序位置对应原先视频中的长度，例如当前拓展后的时序长度为18000，则当前序列中每个位置对应
    # 视频中的0.0094秒
    t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)  
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)  # 聚合连续的索引区间
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])  # 0.0055  内部区间内的每个时序位置分类得分和求平均得到当前区间的类别得分
                # 将当前的到的proposal长度拓展1/4
                len_proposal = len(grouped_temp_list[j])  # 19
                outer_s = max(0, int(grouped_temp_list[j][0] - lambda_ * len_proposal))  # 17976
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + lambda_ * len_proposal))  # 17980
                # [17976,17977,17978,17979,17980]
                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])  # 0.0 外部区间的类别得分
                # 内部的得分减去外部的得分得到当前提议的得分
                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]  # 0.0055
                t_start = grouped_temp_list[j][0] * t_factor   # t_factor : 0.0094
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])  # [5,0.0055,169.42,169.6]
        temp.append(c_temp)
    return temp


def result2json(result):
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': config.class_dict[result[i][j][0]], 'score': result[i][j][1],
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
    return result_file

# 聚合连续的区间
def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def save_best_record_thumos(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("Test_acc: {:.2f}\n".format(test_info["test_acc"][-1]))
    fo.write("average_mAP: {:.4f}\n".format(test_info["average_mAP"][-1]))
    
    tIoU_thresh = np.linspace(0.1, 0.9, 9)
    for i in range(len(tIoU_thresh)):
        fo.write("mAP@{:.1f}: {:.4f}\n".format(tIoU_thresh[i], test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))

    fo.close()
  

def minmax_norm(act_map):
    max_val = nn.ReLU()(torch.max(act_map, dim=1)[0])   # (B, C+1)
    min_val = nn.ReLU()(torch.min(act_map, dim=1)[0])   # (B, C+1)
    delta = max_val - min_val
    delta[delta <=0] = 1
    ret = (act_map - min_val) / delta  # (B,T,C+1)

    return ret


def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    return keep


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    
    
