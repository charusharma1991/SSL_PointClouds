import pickle
import os
import random
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean,cdist
from collections import OrderedDict
import sys
def ball_vec(tree,nodes,pts):
    ball = np.zeros((len(tree.keys()),pts.shape[1]))
    for i in range(len(tree.keys())):
        key = nodes[i]
        
        items = tree[key]
        centre = np.empty([0,pts.shape[1]])
        for j in range(1,len(tree[key])):
            if type(items[j])==str:
                idx = nodes.index(items[j])
                centre = np.concatenate((centre, np.array([ball[int(idx),:]])),axis=0)
            elif items[j]>=0 and items[j]<pts.shape[0]:
                centre = np.concatenate((centre, np.array([pts[int(items[j]),:]])), axis=0)
        if centre.shape[0]>0:
            ball[i,:] = np.mean(centre,axis=0)
    return ball

def find_quad(P,C):
    quadrants = np.zeros((P.shape[0],),dtype=int)
    for i in range(P.shape[0]):
        x = C[i,0]
        y = C[i,1]
        z = C[i,2]
        cx = P[i,0]
        cy = P[i,1]
        cz = P[i,2]

        if (x >= cx and y >= cy and z >= cz ) or (x < cx and y >= cy and z >= cz) :
            quadrants[i] = 1
        elif (x < cx and y < cy and z >= cz) or (x >= cx and y < cy and z >= cz):
            quadrants[i] = 2
        elif (x >= cx and y >= cy and z < cz) or (x < cx and y >= cy and z < cz):
            quadrants[i] = 3
        elif (x < cx and y < cy and z < cz) or (x >= cx and y < cy and z < cz):
            quadrants[i] = 4

    return quadrants

def rev_dict(tree,nodes):
    rev_tree={}
    for i in range(len(tree.keys())):
        rev_tree[nodes[i]]=[]
    for i in range(len(tree.keys())-1,0,-1):
        ball = nodes[i]
        children = tree[ball]
        for j in range(1, len(children)):
            if children[j] in tree.keys():
                rev_tree[children[j]].append(ball)

    rev_tree = OrderedDict(sorted(rev_tree.items(), key=lambda t: t[0]))
    return rev_tree
def get_pairs(rev_tree,tree,nodes,pts,path1):
    diff_pair_quad = np.empty([0,2],dtype=int)
    same_pair_dist = np.empty([0,2],dtype=int)
    same_pair_label = np.empty([0,],dtype=int)
    for i in range(len(tree.keys())):
        ball = nodes[i]
        children = tree[ball]
        if len(children)>1:
            for j in range(1, len(children)):
                if children[j] in tree.keys() and abs(children[0]-tree[children[j]][0])==1:
                    p = nodes.index(ball)#i
                    c = nodes.index(children[j])
                    diff_pair_quad = np.concatenate((diff_pair_quad,np.array([[p,c]])),axis=0)
    for i in range(len(tree.keys())):
        ball = nodes[i]
        children = tree[ball]
        count=i+1
        while count<len(tree.keys()) and children[0]-tree[nodes[count]][0]==0:
            p = nodes.index(ball)#i
            c = nodes.index(nodes[count])#count
            same_pair_dist = np.concatenate((same_pair_dist,np.array([[p,c]])),axis=0)
            if bool(set(rev_tree[ball]) & set(rev_tree[nodes[count]])):
                same_pair_label = np.concatenate((same_pair_label,np.array([1])),axis=None)
            else:
                same_pair_label = np.concatenate((same_pair_label,np.array([0])),axis=None)
            count=count+1
    balls = ball_vec(tree,nodes,pts)
    distance_same_pair = np.sqrt(((balls[same_pair_dist[:,0],:]-balls[same_pair_dist[:,1],:])**2).sum(axis=1))
    diff_pair_label = find_quad(balls[diff_pair_quad[:,0],:],balls[diff_pair_quad[:,1],:])
    if not os.path.exists(path1):
                os.mkdir(path1)
    with open(path1+'/ball_pair_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([diff_pair_quad, diff_pair_label, same_pair_dist, distance_same_pair, same_pair_label, len(nodes)], f)
  
def get_ball_indexes(tree,nodes,adj):
    tree_ids=tree.copy()

    for i in range(len(tree.keys())):
        key = nodes[i]
        items = tree[key]
        ids=[items[0]]
        if len(items)>1:
            for j in range(1, len(items)):
                if type(items[j])==str:
                    ids.extend(tree_ids[items[j]][1:])
                else:
                    ids.append(items[j])
            tree_ids[nodes[i]] = ids
    print(tree)
    print(tree_ids)
    exit()


'''
def map_tree(dataset):
    path = "/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"
    files = [name for name in os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/") if os.path.isdir(os.path.join(path, name))]
    print(files)
    #exit()
    for name in files[20:30]:
        print(name)
        points = os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points_norm/")
        #print(points)
        for pc in points:
            pts = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points_norm/"+pc)
            #balls = os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/sydney/"+name+"/balls/"+pc[:-4])
            f = open("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/dict_adj.pkl", 'rb')
            [tree,nodes,adj,_] = pickle.load(f)
            f.close()
            #get_ball_indexes(tree,nodes,adj)
            #print(tree.keys(),nodes)
            rev_tree = rev_dict(tree,nodes)
            get_pairs(rev_tree,tree,nodes,pts,"/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4])
            #print(balls)

#dataset = sys.argv[1]
#map_tree(dataset)
'''
