import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DistanceNet(nn.Module):
    def __init__(self,args):
        super(DistanceNet, self).__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64,128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128,args.emb_dims, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout()
        self.fc1 = nn.Linear(2*args.emb_dims, 1)

    def forward(self, input1, input2):  # input x = batch,3,N
        x1 = self.conv1(input1.contiguous())
        x1 = self.conv2(x1.contiguous())
        x1 = self.conv3(x1.contiguous())
        x2 = self.conv1(input2.contiguous())
        x2 = self.conv2(x2.contiguous())
        x2 = self.conv3(x2.contiguous())
        x = torch.cat((x1,x2),1).squeeze(2)
        x = self.dp1(x)
        x = self.fc1(x)
        return x

class QuadrantNet(nn.Module):
    def __init__(self,args,output_channels=4):
        super(QuadrantNet, self).__init__()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128,args.emb_dims, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout()
        self.fc1 = nn.Linear(2*args.emb_dims, output_channels)

    def forward(self, input1, input2):  # input x = batch,3,N

        x1 = self.conv1(input1.contiguous())
        x1 = self.conv2(x1.contiguous())
        x1 = self.conv3(x1.contiguous())

        x2 = self.conv1(input2.contiguous())
        x2 = self.conv2(x2.contiguous())
        x2 = self.conv3(x2.contiguous())
        x = torch.cat((x1,x2),1).squeeze(2)
        x = self.dp1(x)
        x = self.fc1(x)
        return x

class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv1 = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(32,64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64,128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.fc1 = nn.Linear(64, 128)

    def forward(self, x):
        x1 = self.conv1(x)
        assert not torch.isnan(x1).any()
        x1 = self.conv2(x1)
        assert not torch.isnan(x1).any()
        x1 = self.conv3(x1)
        assert not torch.isnan(x1).any()
        return x1

def block_diag(A,B):
    r_1 = torch.cat((A, torch.zeros((A.size()[0], B.size()[1])).double().cuda()), dim = 1)
    r_2= torch.cat((torch.zeros((B.size()[0], A.size()[1])).double().cuda(), B), dim = 1)
    C = torch.cat((r_1, r_2), dim = 0)
    return C

def pool_balls(feat,trees,balls,adjacency):
    ball_vec = torch.zeros(size=(0,feat.size()[2])).cuda()
    for i in range(feat.size()[0]):
        x = feat[i]
        tree = trees[i]
        nodes = balls[i]
        adj = torch.tensor(adjacency[i]).cuda()
        ball = torch.zeros(size=(adj.size()[0],x.size()[1])).cuda()
        for j in range(len(tree.keys())):
            items = tree[nodes[j]][1:]
            centre = x[items,:]
            assert not torch.isnan(centre).any()
            ball[j,:] = torch.mean(centre,dim=0)
            assert not torch.isnan(ball).any()
        ball_vec= torch.cat((ball_vec, ball), 0)
        assert not torch.isnan(ball_vec).any()
        if i>0:
            adj = adj+adj.T
            adj[adj>0]=1
            A = block_diag(A,adj)
        else:
            adj = adj+adj.T
            adj[adj>0]=1
            A = adj
    return ball_vec.cuda(),A.cuda()


class CoverNet(nn.Module):
    def __init__(self,args):
        super(CoverNet, self).__init__()
        self.args = args
        self.base = PointNet(args).cuda()
        self.Quad = QuadrantNet(args)
        self.Dist = DistanceNet(args)

    def forward(self, x):  # input x = batch,3,N
        assert not torch.isnan(x).any()
        x = self.base(x)  #x-> batch X N X 128
        return x
