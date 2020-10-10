import pickle
import numpy as np
import os
from collections import OrderedDict
from scipy import sparse
import torch
import sys
import pickle
def filter_ball(dataset,dic,name,pc,tree,nodes,tree_ids,diff_pair_quad,diff_pair_label,same_pair_dist,distance_same_pair,same_pair_label,dir_path):
    diff=2
    queue=[]
    new_nodes=[]
    top_level = int(tree[nodes[len(tree.keys())-1]][0])
    idx_key=np.empty([0,],dtype=int)
    for i in range(len(tree.keys())-1,-1,-1):
        key = nodes[i]
        level = int(tree[key][0])
        if top_level-level<=diff:
            new_nodes.append(key)
            idx_key = np.concatenate((idx_key,i),axis=None)
        elif top_level-level==diff+1 and len(new_nodes)<20:
            diff=diff+1
            new_nodes.append(key)
            idx_key = np.concatenate((idx_key,i),axis=None)
    idx_key = np.sort(idx_key)
    new_nodes = new_nodes[::-1]
    new_tree = tree.copy()
    new_tree_ids = tree_ids.copy()
    for i in range(0,len(tree.keys())):
        key = nodes[i]
        if key not in new_nodes:
            del new_tree[key]
            del new_tree_ids[key]
    diff_level_ids = np.empty([0,2],dtype=int)
    diff_level_label=np.empty([0,],dtype=int)
    same_level_label=np.empty([0,],dtype=int)
    same_level_dist=np.empty([0,])
    same_level_ids = np.empty([0,2],dtype=int)
    for i in range(diff_pair_quad.shape[0]):
        if diff_pair_quad[i,0] in idx_key and diff_pair_quad[i,1] in idx_key:
            id1 = diff_pair_quad[i,0]
            id2 = diff_pair_quad[i,1]
            key1 = nodes[id1]
            key2 = nodes[id2]
            idx1=new_nodes.index(key1)
            idx2=new_nodes.index(key2)
            diff_level_ids = np.concatenate((diff_level_ids,np.array([[idx1,idx2]])),axis=0)
            diff_level_label = np.concatenate((diff_level_label,diff_pair_label[i]),axis=None)
    for i in range(same_pair_dist.shape[0]):
        if same_pair_dist[i,0] in idx_key and same_pair_dist[i,1] in idx_key:
            id1 = same_pair_dist[i,0]
            id2 = same_pair_dist[i,1]
            key1 = nodes[id1]
            key2 = nodes[id2]
            idx1=new_nodes.index(key1)
            idx2=new_nodes.index(key2)
            same_level_ids = np.concatenate((same_level_ids,np.array([[idx1,idx2]])),axis=0)
            same_level_label = np.concatenate((same_level_label,same_pair_label[i]),axis=None)
            same_level_dist = np.concatenate((same_level_dist,distance_same_pair[i]),axis=None)
    ball_count = len(new_nodes)
    if not os.path.exists(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]):
        os.mkdir(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4])
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/tree.npy", new_tree)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/nodes.npy", new_nodes)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/tree_ids.npy", new_tree_ids)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/diff_pair_quad.npy", diff_level_ids)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/diff_pair_label.npy", diff_level_label)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/same_pair_dist.npy", same_level_ids)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/distance_same_pair.npy", same_level_dist)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/same_pair_label.npy", same_level_label)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/ball_count.npy", ball_count)


'''
def map_tree(dataset):
    path = "/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"
    files = [name for name in os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/") if os.path.isdir(os.path.join(path, name))]
    print(files)

    for name in files:
        print(name)
        points = os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points/")
        #print(points)
        for pc in points:
            #print(pc)
            pts = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points/"+pc)
            #print(pts.max())
            #np.save("/home/csharma/PC_TDA/self-supervised/CoverNet/MN10/"+name+"/points/"+pc,pts)
            f = open("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/dict_adj.pkl", 'rb')
            [tree,nodes,_,tree_ids] = pickle.load(f)
            f.close()
            #ball_padded=np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/ball_padded.npy")
            adj = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/adj.npy").tolist().todense()
            #print(adj.shape)
            diff_pair_quad = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/diff_pair_quad.npy")
            diff_pair_label = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/diff_pair_label.npy")
            same_pair_dist = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/same_pair_dist.npy")
            distance_same_pair = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/distance_same_pair.npy")
            same_pair_label = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/same_pair_label.npy")
            ball_count = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4]+"/ball_count.npy")
            filter_ball(dataset, name,pc,tree,nodes,adj,tree_ids,diff_pair_quad,diff_pair_label,same_pair_dist,distance_same_pair,same_pair_label)
            #exit()
            #balls = os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/balls/"+pc[:-4])
            #make_dict(pts,balls,"/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/balls/"+pc[:-4]+"/", "/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4])
            #print(balls)

#dataset = sys.argv[1]
#map_tree(dataset)

path = "/home/csharma/PC_TDA/self-supervised/CoverNet/ModelNet10_512/"
files = [name for name in os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/ModelNet10_512/") if os.path.isdir(os.path.join(path, name))]
print(files)
for name in files:
    if not os.path.exists("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/"):
                os.mkdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/")

for name in files:
    if not os.path.exists("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points_scale/"):
                #os.mkdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points/")
                #os.mkdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/balls/")
                #os.mkdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/")
                os.mkdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points_scale/")
print(files)
'''
