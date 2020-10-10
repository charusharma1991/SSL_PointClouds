#!/usr/bin/env python
#
# File: test_covertree.py
# Date of creation: 11/20/08
# Copyright (c) 2007, Thomas Kollar <tkollar@csail.mit.edu>
# Copyright (c) 2011, Nil Geisweiller <ngeiswei@gmail.com>
# All rights reserved.
#
# This is a tester for the cover tree nearest neighbor algorithm.  For
# more information please refer to the technical report entitled "Fast
# Nearest Neighbors" by Thomas Kollar or to "Cover Trees for Nearest
# Neighbor" by John Langford, Sham Kakade and Alina Beygelzimer
#  
# If you use this code in your research, kindly refer to the technical
# report.
import pickle
from ball_pair_data import get_pairs,rev_dict
from dict_covertree import make_dict
from ball_preprocess import filter_ball
from covertree import CoverTree
from naiveknn import knn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, scale,MinMaxScaler
from numpy import subtract, dot, sqrt
import numpy.random as rd
import time
import os
import operator
from scipy.spatial.distance import cdist, euclidean
import numpy as np
import shutil
from sklearn import metrics
import sys

# Normalize the 3D points
def getData(mat):
    choice = np.random.choice(mat.shape[0], 100, replace=False)
    mat = mat[choice, :]
    norm = normalize(mat, norm='l2')
    return norm

def distance(p, q):
    # print "distance"
    # print "p =", p
    # print "q =", q
    x = subtract(p, q)
    return sqrt(dot(x, x))

# Generate covertree and ball pairs for self-supervised labels
def generate_covertree(bas):
    dataset = 'Sydney10'
    dic = 'dict'
    dir_path = "../" #path to dataset
    path = dir_path+dataset+"/"
    files = [name for name in os.listdir(dir_path+dataset+"/") if os.path.isdir(os.path.join(path, name))]
    print(files)
    for name in files:
        print(name)
        names = os.listdir(dir_path+dataset+"/"+name+"/points_norm/")
        #choice = np.random.choice(len(points), len(points), replace=False)
        #print(points)
        #points = [points[c] for c in choice]
        for pc in names:
            #print(pc)
            pts = np.load(dir_path+dataset+"/"+name+"/points_norm/"+pc)
            #pts = getData(pts) #uncomment this to normalize the original data points used for creating cover tree and in SSL network
            #np.save('CoverNet/sydney/'+name+'/points/'+pc[:-4]+'.npy',pts)
            ct = CoverTree(distance,bas)
            for p in pts:
                ct.insert(p)
            if not os.path.exists(dir_path+dataset+"/"+name+"/"+dic):
                os.mkdir(dir_path+dataset+"/"+name+"/"+dic)
            if os.path.exists(dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4]):
                shutil.rmtree(dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4])
            if not os.path.exists(dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4]):
                os.mkdir(dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4])
            ct.writeDotty(dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4]+"/")
            balls = os.listdir(dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4])
            make_dict(pts,balls,dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4]+"/", dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4])
            f = open(dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4]+"/dict_adj.pkl", 'rb')
            [tree,nodes,tree_ids] = pickle.load(f)
            rev_tree = rev_dict(tree,nodes)
            get_pairs(rev_tree,tree,nodes,pts,dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4])
            f = open(dir_path+dataset+"/"+name+"/"+dic+'/'+pc[:-4]+"/ball_pair_data.pkl", 'rb')
            [diff_pair_quad, diff_pair_label, same_pair_dist, distance_same_pair, same_pair_label, ball_count] = pickle.load(f)
            filter_ball(dataset,dic,name,pc,tree,nodes,tree_ids,diff_pair_quad,diff_pair_label,same_pair_dist,distance_same_pair,same_pair_label,dir_path)

if __name__ == '__main__':
    bas = float(sys.argv[1])#bas is base of the radius, default 2.0
    generate_covertree(bas)
