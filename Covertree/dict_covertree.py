import pickle
import numpy as np
import os
from collections import OrderedDict
from scipy import sparse
import torch
import sys
def get_ball_indexes(tree,nodes):
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
    return tree_ids

def make_dict(pts,balls,path,path1):
    levels = np.array([int(ball[:-4]) for ball in balls])
    levels = np.sort(levels)
    parents=[]
    children=[]
    adjacency=[]
    level=[]
    count=0
    for i in range(levels[0],levels[-1]+1):
        f = open(path+str(i)+'.pkl', 'rb')
        [parent,child] = pickle.load(f)
        f.close()
        p = np.unique(parent,axis=0)
        q = np.unique(child,axis=0)
        adj = np.zeros((p.shape[0],q.shape[0]))
        for j in range(parent.shape[0]):
            p1 = parent[j]
            q1 = child[j]
            u = np.where(np.all(p==p1,axis=1))[0]
            v = np.where(np.all(q==q1,axis=1))[0]
            adj[u,v]=1
        parents.append(p)
        children.append(q)
        adjacency.append(adj)
        level.append(chr(65+count))
        count=count+1
    for i in range(len(parents)):
        p = np.array(parents[i])
        q = np.array(children[i])
        adj = np.array(adjacency[i])
        for j in range(p.shape[0]):
            pt= p[j]
            if np.sum(adj[j,:])>0:
                idxs = np.nonzero(adj[j,:])[0]
                centre = np.mean(q[idxs,:],axis=0)
                if pt in np.array(parents[i]):
                    idx = np.where(np.all(np.array(parents[i])==pt,axis=1))[0]
                    for l in range(len(idx)):
                        parents[i][idx[l],:]=centre
                for k in range(i+1,len(children)):
                    if pt in np.array(children[k]):
                        idx = np.where(np.all(np.array(children[k])==pt,axis=1))[0]
                        for l in range(len(idx)):
                            children[k][idx[l],:]=centre
                    if pt in np.array(parents[k]):
                        idx = np.where(np.all(np.array(parents[k])==pt,axis=1))[0]
                        for l in range(len(idx)):
                            parents[k][idx[l],:]=centre
    tree = {}
    nodes=[]
    pt_nodes=[]
    for i in range(len(parents)):
        p = np.array(parents[i])
        q = np.array(children[i])
        adj = np.array(adjacency[i])
        for j in range(p.shape[0]):
            pt= p[j]
            if np.sum(adj[j,:])>0:
                if level[i]+str(j) not in tree.keys():
                    tree[level[i]+str(j)] = [i]
                    nodes.append(level[i]+str(j))
                idxs = np.nonzero(adj[j,:])[0]
                for k in idxs:
                    if np.sum(np.all(pts==q[k],axis=1))>0:
                        u = np.where(np.all(pts==q[k],axis=1))[0][0]
                        pt_nodes.append(u)
                        tree[level[i]+str(j)].append(u)
                    elif i>0 and np.sum(np.all(np.array(parents[i-1])==q[k],axis=1))>0:
                        u = np.where(np.all(np.array(parents[i-1])==q[k],axis=1))[0][0]
                        if level[i-1]+str(u) in tree.keys() and len(tree[level[i-1]+str(u)])>1:
                            tree[level[i]+str(j)].append(level[i-1]+str(u))
    cp_tree=tree.copy()
    modi_key=[]
    modi_val=[]
    for i in range(len(cp_tree.keys())):
        if len(cp_tree[nodes[i]])<3:
            count=0
            for j in range(len(cp_tree[nodes[i]])):
                if type(cp_tree[nodes[i]][j])!=str:
                    count=count+1
            if count>1 and len(cp_tree[nodes[i]]) == count:
                modi_key.append(nodes[i])
                modi_val.append(cp_tree[nodes[i]][1])
                del tree[nodes[i]]
            if count==1 and len(cp_tree[nodes[i]]) == count:
                del tree[nodes[i]]
    tree = OrderedDict(sorted(tree.items(), key=lambda t: t[0]))
    new_nodes=[]
    for node in nodes:
        if node in tree.keys():
            new_nodes.append(node)

    nodes=new_nodes
    for i in range(len(tree.keys())):
        for j in range(1,len(tree[nodes[i]])):
            if tree[nodes[i]][j] in modi_key:
                idx = modi_key.index(tree[nodes[i]][j])
                tree[nodes[i]][j]=modi_val[idx]
    new_pt_nodes = []
    for k,v in tree.items():
        if len(v)==1:
            print('true')
        new_pt_nodes.extend(v[1:])
    new_pt_nodes = list(set(new_pt_nodes))
    new_ptnodes=[]
    for node in pt_nodes:
        if node in new_pt_nodes:
            new_ptnodes.append(node)
    
    pt_nodes = new_ptnodes
    nodes.extend(list(set(pt_nodes)))
    tree_ids = get_ball_indexes(tree,nodes)
    if not os.path.exists(path1):
                os.mkdir(path1)
    with open(path1+'/dict_adj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([tree, nodes, tree_ids], f)
'''
def map_tree(dataset):
    path = "/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"
    files = [name for name in os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/") if os.path.isdir(os.path.join(path, name))]
    print(files)

    for name in files:
        print(name)
        points = os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points_norm/")
        #print(points)
        for pc in points:
            #print(pc)
            pts = np.load("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/points_norm/"+pc)
            #print(pts.shape)
            #exit()
            balls = os.listdir("/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/balls/"+pc[:-4])
            make_dict(pts,balls,"/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/balls/"+pc[:-4]+"/", "/home/csharma/PC_TDA/self-supervised/CoverNet/"+dataset+"/"+name+"/dict/"+pc[:-4])
            #print(balls)

#dataset = sys.argv[1]
#map_tree(dataset)
'''
