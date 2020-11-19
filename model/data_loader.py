# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

from fragments import calculate_fragments

def compute_fragments(seq_len, action_state_size):
    
    # "action_fragments" contains the starting and ending frame of each action fragment
    frag_jump = calculate_fragments(seq_len, action_state_size)
    action_fragments = torch.zeros((action_state_size,2), dtype=torch.int64)
    for i in range(action_state_size-1):
        action_fragments[i,1] = torch.tensor(sum(frag_jump[0:i+1])-1)
        action_fragments[i+1,0] = torch.tensor(sum(frag_jump[0:i+1]))
    action_fragments[action_state_size-1, 1] = torch.tensor(sum(frag_jump[0:action_state_size])-1)    
                
    return action_fragments

class VideoData(Dataset):
    def __init__(self, mode, split_index, action_state_size):
        self.mode = mode
        self.name = 'tvsum'
        self.datasets = ['../data/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         '../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5']
        self.splits_filename = ['../data/splits/' + self.name + '_splits.json']
        self.split_index = split_index # it represents the current split (varies from 0 to 4)

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        hdf = h5py.File(self.filename, 'r')
        self.action_fragments = {}
        self.list_features = []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i==self.split_index:
                    self.split = split
                    
        for video_name in self.split[self.mode + '_keys']:
            features = torch.Tensor(np.array(hdf[video_name + '/features']))
            self.list_features.append(features)
            self.action_fragments[video_name] = compute_fragments(features.shape[0], action_state_size)

        hdf.close()

    def __len__(self):
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    # In "train" mode it returns the features and the action_fragments; in "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_name = self.split[self.mode + '_keys'][index]  #gets the current video name
        frame_features = self.list_features[index]

        if self.mode == 'test':
            return frame_features, video_name, self.action_fragments[video_name]
        else:
            return frame_features, self.action_fragments[video_name]

def get_loader(mode, split_index, action_state_size):
    if mode.lower() == 'train':
        vd = VideoData(mode, split_index, action_state_size)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, split_index, action_state_size)


if __name__ == '__main__':
    pass
