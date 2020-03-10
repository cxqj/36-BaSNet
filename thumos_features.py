import torch.utils.data as data
import os
import csv
import json
import numpy as np
import torch
import pdb
import time
import random
import utils
import config


class ThumosFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, len_feature, sampling, seed=-1, supervision='weak'):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode  # 'train'/'test'
        self.modal = modal # 'all'
        self.feature_fps = feature_fps  # 25
        self.num_segments = num_segments  # 750
        self.len_feature = len_feature  # 2048

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(data_path, 'features', self.mode, self.modal)

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()

        # 动作类对应的idx
        self.class_name_to_idx = dict((v, k) for k, v in config.class_dict.items())        
        self.num_classes = len(self.class_name_to_idx.keys())  # 20

        self.supervision = supervision  # weak 以弱监督方式运行
        self.sampling = sampling  # 在选取视频帧时的方式，训练时:random,测试时:uniform


    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        # data: 连接后的特征(T,C) vid_num_seg: 真实特征序列长度  sample_idx: 采样的序列索引
        data, vid_num_seg, sample_idx = self.get_data(index)  
        # label：[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]  temp_anno: 0
        label, temp_anno = self.get_label(index, vid_num_seg, sample_idx)

        return data, label, temp_anno, self.vid_list[index], vid_num_seg

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name + '.npy')).astype(np.float32)  #(T,C)

            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])   # 特征采样的索引，别的地方也可以借鉴这个方式
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]   # (750,1024)
            flow_feature = flow_feature[sample_idx] # (750,1024)

            feature = np.concatenate((rgb_feature, flow_feature), axis=1)  # (750,2048)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx

    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]
        #[{'segments':[18.6,24.8],'label':'HighJump'},{'segments':[18.6,24.8],'label':'HighJump'}]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)  # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        classwise_anno = [[]] * self.num_classes

        # 构建one_hot形式的label
        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_anno = np.zeros([vid_num_seg, self.num_classes])
            t_factor = self.feature_fps / 16

            for class_idx in range(self.num_classes):
                if label[class_idx] != 1:
                    continue

                for _anno in classwise_anno[class_idx]:
                    tmp_start_sec = float(_anno['segment'][0])
                    tmp_end_sec = float(_anno['segment'][1])

                    tmp_start = round(tmp_start_sec * t_factor)
                    tmp_end = round(tmp_end_sec * t_factor)

                    temp_anno[tmp_start:tmp_end+1, class_idx] = 1

            temp_anno = temp_anno[sample_idx, :]

            return label, torch.from_numpy(temp_anno)

    # 随机采样的方式，生成750个索引(有些时间位置可以重复)
    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)  # 例如：[0,0,0,1,1,2,2,.....,480] shape：750


    def uniform_sampling(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)




