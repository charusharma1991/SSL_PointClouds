from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import pickle
from sklearn.preprocessing import normalize, scale,MinMaxScaler

class FSL_sampler(data.Dataset):
    def __init__(self, data,labels,num_label,npoints,root):
        self.root = root
        self.npoints = npoints
        self.data = data
        self.labels=labels
        self.num_label=num_label
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        self.meta = []
        for i,lb in enumerate(self.labels):
            dir_point = os.path.join(self.root, lb,'points_norm')
            dir_dict = os.path.join(self.root, lb, 'dict')
            self.meta.append((os.path.join(dir_point, self.data[i][:-4] + '.npy'), os.path.join(dir_dict,self.data[i][:-4]+"/"),os.path.join(dir_dict,self.data[i][:-4]+"/")))
        self.datapath = []
        for i,fn in enumerate(self.meta):
            self.datapath.append((self.num_label[i], fn[0], fn[1], fn[2]))            


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.load(fn[1]).astype(np.float32)
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        point_set = point_set[choice, :]
        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set,fn[2],fn[3],cls
        
    def __len__(self):
        return len(self.datapath)

'''
if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'CATH_data', class_choice = ['A'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'CATH_data', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
'''
