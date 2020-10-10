#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_classifier import FSL_sampler
from model_classifier import PointNet, DGCNN,PointNetCls
from model_conv_classifier import CoverNet
import numpy as np
from torch.utils.data import DataLoader
from utils import cal_loss, IOStream
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
matplotlib.use('Agg')
import pickle

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

def train(args, io):
    f = open("../"+args.split_path+'.pkl', 'rb')
    [data_train,label_train,tr_label, data_test, label_test,te_label] = pickle.load(f)
    f.close()
    train_dataset = FSL_sampler(data_train,label_train,tr_label,npoints = args.num_points,root = args.dataset_name)
    test_dataset = FSL_sampler(data_test,label_test,te_label,npoints = args.num_points,root = args.dataset_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=4,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=True, num_workers=4,drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.emb_dims=256
    model1 = CoverNet(args).to(device)
    #Try to load models
    args.emb_dims=1024
    if args.model == 'pointnet':
        model = PointNetCls(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'pointcnn':
        model = Classifier(args).to(device)
    else:
        raise Exception("Not implemented")
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    checkpoint = torch.load(args.model_path)
    args.emb_dims=256
    model1.load_state_dict(checkpoint['state_dict'])
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss
    best_test_acc = 0
    best_sil=-1
    best_sil1=-1
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        if True:
        #with torch.autograd.set_detect_anomaly(True):
            model.train()
            train_pred = []
            train_true = []
            for data, label in train_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                args.emb_dims=256
                data1 = model1(data)
                args.emb_dims=1024
                logits,_ = model(data1)
                loss = criterion(logits, label)
                loss.backward()
                opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            args.emb_dims=256
            data1 = model1(data)
            args.emb_dims=1024
            logits,_ = model(data1)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.pth' % args.exp_name)
    io.cprint('best test acc: %.6f'% (best_test_acc))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='Sydney10_FSL', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset_name', type=str, default='../Sydney10', metavar='N'
                       )
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=100,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--test_K_way', type=int, default=10,
            help='Number of classes for doing each classification run')
    parser.add_argument('--train_K_way', type=int, default=10,
            help='Number of classes for doing each training comparison')
    parser.add_argument('--test_N_shots', type=int, default=20,
            help='Number of shots in test')
    parser.add_argument('--train_N_shots', type=int, default=20,
            help='Number of shots when training')
    parser.add_argument('--model_path', type=str, default='../SSL/cls_Sydney10/model_3.pth', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--split_path', type=str, default='data_sydney10/train_1020_test_1020_1', metavar='N',
			help='output folder')
    parser.add_argument('--no', type=int, default=1, 
			help='sr no')
    args = parser.parse_args()

    _init_()
    
    io = IOStream('checkpoints/' + args.exp_name + '/run'+str(args.train_K_way)+'_'+str(args.train_N_shots)+'_'+str(args.num_points)+"_"+str(args.no)+'.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    train(args, io)
