from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from data_FSL import FSL_sampler
from model_conv import CoverNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import cal_loss, IOStream,dist_loss
import sklearn.metrics as metrics
import pickle
import time
import scipy.io as sio
def get_dict(files):
    trees = []
    balls = []
    for i in range(len(files)):
        tree_ids = np.load(files[i]+"tree_ids.npy",allow_pickle='TRUE').item()
        nodes = np.load(files[i]+"nodes.npy")
        trees.append(tree_ids)
        balls.append(nodes)
    return trees,balls
def get_pairs(trees, files):
    diff_level_label=np.empty([0,],dtype=int)
    same_level_label=np.empty([0,],dtype=int)
    same_level_dist=np.empty([0,])
    diff_level_ids = np.empty([0,2],dtype=int)
    same_level_ids = np.empty([0,2],dtype=int)
    count=0
    for i in range(len(files)):
        diff_pair_quad = np.load(files[i]+"diff_pair_quad.npy")
        diff_pair_label = np.load(files[i]+"diff_pair_label.npy")
        same_pair_dist = np.load(files[i]+"same_pair_dist.npy")
        distance_same_pair = np.load(files[i]+"distance_same_pair.npy")
        same_pair_label = np.load(files[i]+"same_pair_label.npy")
        ball_count = np.load(files[i]+"ball_count.npy")
        idxs1 = np.nonzero(same_pair_label)[0]
        idxs0 = np.where(same_pair_label == 0)[0]
        idxs = np.concatenate((idxs1,idxs0[0:len(idxs1)]),axis=None)
        same_pair_label = same_pair_label[idxs]
        same_pair_dist = same_pair_dist[idxs,:]
        distance_same_pair = distance_same_pair[idxs]
        diff_level_label = np.concatenate((diff_level_label,diff_pair_label),axis=None)
        same_level_label = np.concatenate((same_level_label,same_pair_label),axis=None)
        same_level_dist = np.concatenate((same_level_dist,distance_same_pair),axis=None)
        diff_pair_quad = diff_pair_quad + count
        same_pair_dist = same_pair_dist + count
        diff_level_ids = np.concatenate((diff_level_ids,diff_pair_quad),axis=0)
        same_level_ids = np.concatenate((same_level_ids,same_pair_dist),axis=0)
        count=count+ball_count
    return torch.from_numpy(diff_level_ids.astype(np.int64)), torch.from_numpy(diff_level_label.astype(np.int64)), torch.from_numpy(same_level_ids.astype(np.int64)), torch.from_numpy(same_level_dist), torch.from_numpy(same_level_label.astype(np.int64))

def train(args):
    f = open("../"+args.split_path+'.pkl', 'rb')
    [data_train,label_train,tr_label, data_test, label_test,te_label] = pickle.load(f)
    f.close()
    train_dataset = FSL_sampler(data_train,label_train,tr_label,npoints = args.num_points,root = args.dataset_name)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=4,drop_last=False)
    test_dataset = FSL_sampler(data_test,label_test,te_label,npoints = args.num_points,root = args.dataset_name)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,shuffle=True, num_workers=4,drop_last=False)
    print(len(train_dataset), len(test_dataset))
    num_classes = len(train_dataset.classes)
    print('classes', num_classes)
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model == 'CoverNet':
        model = CoverNet(args).to(device)
    else:
        raise Exception("Not implemented")
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd==1:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-6)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    start_epoch=0
    if args.model_path != '':
        checkpoint = torch.load(args.model_path)
        opt.load_state_dict(checkpoint['opt'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        loss = checkpoint['loss']
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    regress = dist_loss
    for epoch in range(start_epoch,args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []
        if True:
            elapsed=0
        #with torch.autograd.set_detect_anomaly(True):
            model.train()
            for i, data in enumerate(train_dataloader, 0):
                points,names,balls_pair, target = data
                trees,balls = get_dict(names)
                diff_level_ids, diff_level_label, same_level_ids, same_level_dist, same_level_label = get_pairs(trees, balls_pair)#get SSL labels
                points, target = Variable(points), Variable(target[:,0])
                diff_level_label = Variable(diff_level_label).cuda()
                same_level_label = Variable(same_level_label).cuda()
                same_level_dist = Variable(same_level_dist).cuda()
                diff_level_ids = Variable(diff_level_ids).cuda()
                same_level_ids = Variable(same_level_ids).cuda()
                batch_size = points.size()[0]
                points = points.permute(0, 2, 1)
                points, target = points.cuda(), target.cuda()
                opt.zero_grad()
                start = time.time()
                Quad, Dist = model(points,trees,balls,diff_level_ids,same_level_ids)#model
                elapsed = elapsed + (time.time() - start)
                Quad_loss = criterion(Quad.double(), diff_level_label-1)
                distance_loss = regress(Dist.double(),same_level_dist )
                loss = Quad_loss+distance_loss
                assert not torch.isnan(loss).any()
                loss.backward()
                opt.step()
                count += batch_size
                train_loss += loss.item() * batch_size
            outstr = 'Train epoch %d, loss: %.6f, time: %6f' % (epoch,train_loss*1.0/count,elapsed)
            io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for i, data in enumerate(test_dataloader, 0):
            points,names,balls_pair, target = data
            trees,balls = get_dict(names)
            diff_level_ids, diff_level_label, same_level_ids, same_level_dist, same_level_label = get_pairs(trees, balls_pair)
            points, target = Variable(points), Variable(target[:,0])
            diff_level_label = Variable(diff_level_label).cuda()
            same_level_label = Variable(same_level_label).cuda()
            same_level_dist = Variable(same_level_dist).cuda()
            diff_level_ids = Variable(diff_level_ids).cuda()
            same_level_ids = Variable(same_level_ids).cuda()
            batch_size = points.size()[0]
            points = points.permute(0, 2, 1)
            points, target = points.cuda(), target.cuda()
            Quad, Dist = model(points,trees,balls,diff_level_ids,same_level_ids)
            Quad_loss = criterion(Quad.double(), diff_level_label-1)
            distance_loss = regress(Dist.double(),same_level_dist )
            loss = Quad_loss+distance_loss
            count += batch_size
            test_loss += loss.item() * batch_size
        outstr = 'Test %d, loss: %.6f' % (epoch,test_loss*1.0/count)
        io.cprint(outstr)
        if epoch>=0:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),'opt': opt.state_dict(), 'loss': loss, }
            torch.save(state, '%s/model_%d.pth' % (args.outf, epoch))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Self-supervised Point Cloud')
    parser.add_argument('--exp_name', type=str, default='Sydney10_SSL', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='CoverNet', metavar='N',
                        choices=['CoverNet'],
                        help='Model to use, [CoverNet]')
    parser.add_argument('--dataset_name', type=str, default='../Sydney10', metavar='N',
                        )
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=int, default=0,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=100,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=256, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--test_K_way', type=int, default=10,
            help='Number of classes for doing each classification run')
    parser.add_argument('--train_K_way', type=int, default=10,
            help='Number of classes for doing each training comparison')
    parser.add_argument('--test_N_shots', type=int, default=20,
            help='Number of shots in test')
    parser.add_argument('--train_N_shots', type=int, default=20,
            help='Number of shots when training')
    parser.add_argument('--model_path', type=str, default='', metavar='N', 
			choices=['','cls_sydney/model_199.pth','cls_ModelNet10_2/model_15.pth'],
                        help='Pretrained model path')
    parser.add_argument('--outf', type=str, default='cls_Sydney10', metavar='N',
			help='output folder')
    parser.add_argument('--split_path', type=str, default='data_sydney10/train_1020_test_1020_1', metavar='N',
			help='output folder')
    args = parser.parse_args()
    io = IOStream(args.outf + '/run.log')
    io.cprint(str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    train(args)
