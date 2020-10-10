# File: covertree.py
# Date of creation: 05/04/07
# Copyright (c) 2007, Thomas Kollar <tkollar@csail.mit.edu>
# Copyright (c) 2011, Nil Geisweiller <ngeiswei@gmail.com>
# All rights reserved.
#
# This is a class for the cover tree nearest neighbor algorithm.  For
# more information please refer to the technical report entitled "Fast
# Nearest Neighbors" by Thomas Kollar or to "Cover Trees for Nearest
# Neighbor" by John Langford, Sham Kakade and Alina Beygelzimer
#  
# If you use this code in your research, kindly refer to the technical
# report.

import numpy as np
import operator
from random import choice
from heapq import nsmallest, heappush, heappop
from itertools import product
from collections import Counter
import pickle
'''
try:
    from collections import Counter
except ImportError: # Counter is not available in Python before v2.7
    from recipe_576611_1 import Counter
try:
    from joblib import Parallel, delayed
except ImportError:
    pass
import cStringIO
'''

# method that returns true iff only one element of the container is True
def unique(container):
    return Counter(container).get(True, 0) == 1


# the Node representation of the data
class Node:
    # data is an array of values
    def __init__(self, data=None, idx=None):
        self.data = data
        self.children = {}      # dict mapping level and children
        self.parent = None
        self.idx = idx

    # addChild adds a child to a particular Node and a given level i
    def addChild(self, child, i):
        try:
            # in case i is not in self.children yet
            if(child not in self.children[i]):
                self.children[i].append(child)
        except(KeyError):
            self.children[i] = [child]
        child.parent = self

    # getChildren gets the children of a Node at a particular level
    def getChildren(self, level):
        retLst = [self]
        try:
            retLst.extend(self.children[level])
        except(KeyError):
            pass
        
        return retLst

    # like getChildren but does not return the parent
    def getOnlyChildren(self, level):
        try:
            return self.children[level]
        except(KeyError):
            pass
        
        return []


    def removeConnections(self, level):
        if(self.parent != None):
            self.parent.children[level+1].remove(self)
            self.parent = None

    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return str(self.data)

class CoverTree:
    
    #
    # Overview: initalization method
    #
    # Input: distance function, root, maxlevel, minlevel, base, and
    #  for parallel support jobs and min_len_parallel. Here root is a
    #  point, maxlevel is the largest number that we care about
    #  (e.g. base^(maxlevel) should be our maximum number), just as
    #  base^(minlevel) should be the minimum distance between Nodes.
    #
    #  In case parallel is enabled (jobs > 1), min_len_parallel is the
    #  minimum number of elements at a given level to have their
    #  distances to the element to insert or query evaluated.
    #
    def __init__(self, distance,bas, data=None, root = None, maxlevel = 3, base = 2.0,
                 jobs = 1, min_len_parallel = 100):
        self.distance = distance
        self.root = root
        self.maxlevel = maxlevel
        self.minlevel = maxlevel # the minlevel will adjust automatically
        self.idx = 0
        self.base = bas
        self.jobs = jobs
        self.min_len_parallel = min_len_parallel
        # for printDotty
        self.__printHash__ = set()

        if data is None:
            data = []

        for point in data:
            self.insert(point)


    @property
    def size(self):
        "Number of elements in the tree"
        return self.idx


    #
    # Overview: insert an element p into the tree
    #
    # Input: p
    # Output: nothing
    #
    def insert(self, p):
        if self.root == None:
            self.root = self._newNode(p)
        else:
            self._insert_iter(p)

    def _newNode(self, *args, **kws):
        kws['idx'] = self.idx
        self.idx += 1
        return Node(*args, **kws)

    def __iter__(self):
        """
        Breadth-first traversal of the nodes in the tree
        Output:
          - iterable of (idx, point)
        """

        queue = [(self.maxlevel, self.root)]

        observed = set()

        while queue:
            lvl, node = queue.pop(0)
            if node not in observed:
                yield node.idx, node.data

            observed.add(node)

            next_lvl = lvl - 1
            if next_lvl < self.minlevel: continue

            for child in node.getChildren(next_lvl):
                queue.append((next_lvl, child))

    def extend(self, iterable):
        if isinstance(iterable, CoverTree):
            getter = operator.itemgetter(1)
        else:
            getter = lambda x: x
        for p in map(getter, iterable):
            self.insert(p)

    #
    # Overview:insert an element p in to the cover tree
    #
    # Input: point p
    #
    # Output: nothing
    #
    def _insert_iter(self, p):
        Qi_p_ds = [(self.root, self.distance(p, self.root.data))]
        i = self.maxlevel
        while True:
            # get the children of the current level
            # and the distance of the all children
            Q_p_ds = self._getChildrenDist_(p, Qi_p_ds, i)
            d_p_Q = self._min_ds_(Q_p_ds)

            if d_p_Q == 0.0:    # already there, no need to insert
                return
            elif d_p_Q > self.base**i: # the found parent should be right
                break
            else: # d_p_Q <= self.base**i, keep iterating

                # find parent
                if self._min_ds_(Qi_p_ds) <= self.base**i:
                    parent = choice([q for q, d in Qi_p_ds if d <= self.base**i])
                    pi = i
                
                # construct Q_i-1
                Qi_p_ds = [(q, d) for q, d in Q_p_ds if d <= self.base**i]
                i -= 1

        # insert p
        parent.addChild(self._newNode(p), pi)
        # update self.minlevel
        self.minlevel = min(self.minlevel, pi-1)


    def neighbors(self, point, radius):
        """
        Overview: get the neighbors of `p` within distance `r`

        Input:
         - point :: a point
         - radius :: float - the maximum (inclusive) distance
        Output:
         - [(i, n, d)] :: list of pairs (`index`, `point`, `float`) which are the point and it's distance to `p`
        """

        def containsPoint(point, radius, node, level, dist=None):
            if dist is None:
                dist = self.distance(point, node.data)
            # print level, point, dist, radius, radius + self.base**level
            return dist <= radius + self.base**level


        if self.root is None:
            return []

        result = set()
        queue = [(self.maxlevel, self.root, self.distance(point, self.root.data))]

        while queue:
            level, node, dist = queue.pop(0)

            if not containsPoint(point, radius, node, level, dist=dist):
                continue

            if dist <= radius:
                result.add((node, dist))

            next_level = level-1
            if next_level < self.minlevel: continue

            for child in node.getChildren(next_level):
                if not child == node:
                    d = self.distance(point, child.data)
                else:
                    d = dist
                queue.append((next_level, child, d))


        return map(lambda x: (x[0].idx, x[0].data, x[1]), result)


    def contains(self, point, eps=1.0):
        """
        Ask if the cover tree contains a given point

        Input:
          - point :: the query point  -- the point to search for
          - eps   :: double           -- epsilon for distance comparison

        Output:
          - found :: bool             -- indicates presence of point in Cover Tree
        """

        nn = self.neighbors(point, eps)
        nn = list(nn) # force the lazy calculation

        if len(nn) == 1:
            return True
        elif len(nn) == 0:
            return False
        else: raise(ValueError, 'Found multiple results for {} with eps={}: {}'.format(point, eps, nn))

    def knn(self, p, k):
        """
        Get the `k` nearest neighbors of `point`

        Input:
          - point :: a point
          - k     :: positive int

        Output:
          - [(i, p, d)] :: list of length `k` of the index, point, and distance in the CT closest to input `point`
        """

        Qi_p_ds = [(self.root, self.distance(p, self.root.data))]
        for i in reversed(range(self.minlevel, self.maxlevel+1)):
            Q_p_ds = self._getChildrenDist_(p, Qi_p_ds, i)
            _, d_p_Q = self._kmin_p_ds_(k, Q_p_ds)[-1]
            Qi_p_ds = [(q, d) for q, d in Q_p_ds if d <= d_p_Q + self.base**i]
        res = map(lambda x: (x[0].idx, x[0].data, x[1]), Qi_p_ds)
        return nsmallest(k, res, key=operator.itemgetter(2))


    #
    # Overview: get the children of cover set Qi at level i and the
    # distances of them with point p
    #
    # Input: point p to compare the distance with Qi's children, and
    # Qi_p_ds the distances of all points in Qi with p
    #
    # Output: the children of Qi and the distances of them with point
    # p
    #
    def _getChildrenDist_(self, p, Qi_p_ds, i):
        Q = sum([n.getOnlyChildren(i) for n, _ in Qi_p_ds], [])
        Q_p_ds = [(q, self.distance(p, q.data)) for q in Q]
        return Qi_p_ds + Q_p_ds

    #
    # Overview: get a list of pairs <point, distance> with the k-min distances
    #
    # Input: Input cover set Q, distances of all nodes of Q to some point
    # Output: list of pairs 
    #
    def _kmin_p_ds_(self, k, Q_p_ds):
        return nsmallest(k, Q_p_ds, lambda x: x[1])

    # return the minimum distance of Q_p_ds
    def _min_ds_(self, Q_p_ds):
        return self._kmin_p_ds_(1, Q_p_ds)[0][1]

    # format the final result. If without_distance is True then it
    # returns only a list of data points, other it return a list of
    # pairs <point.data, distance>
    def _result_(self, res, without_distance):
        if without_distance:
            return [p.data for p, _ in res]
        else:
            return [(p.data, d) for p, d in res]
    
    #
    # Overview: write to a file the dot representation
    #
    # Input: None
    # Output: 
    #
    def writeDotty(self,path):
        #outputFile.write("digraph {\n")
        self._writeDotty_rec_new(path,[self.root], self.maxlevel)
        #outputFile.write("}")


    def _writeDotty_rec_new(self,path, C, i):

        if(i == self.minlevel):
            return
        #print(i,self.minlevel,len(C))
        children = []
        count=0
        parent = np.empty([0,3])
        child = np.empty([0,3])
        for p in C:
            childs = p.getChildren(i)
            #print(childs)
            if len(childs)>1 or len(C)>1:
                for q in childs:
                    #print(p,q)
                    #print(p.shape,parent.shape)
                    parent = np.concatenate((parent,np.array([p.data])),axis=0)
                    child = np.concatenate((child,np.array([q.data])),axis=0)
                    count=count+1
            children.extend(childs)
        #print(parent.shape,child.shape)
        if parent.shape[0]>0:
            with open(path+str(i)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([parent, child], f)
        #print(count)
        self._writeDotty_rec_new(path, children, i-1)


    #
    # Overview:recursively build printHash (helper function for writeDotty)
    #
    # Input: C, i is the level
    #
    def _writeDotty_rec(self, outputFile, C, i):

        if(i == self.minlevel):
            return
        print(i,self.minlevel)
        children = []
        for p in C:

            childs = p.getChildren(i)

            for q in childs:
                print(p,q)
                outputFile.write("\"lev:" +str(i) + " "
                                 + str(p.data) + "\"->\"lev:"
                                 + str(i-1) + " "
                                 + str(q.data) + "\"\n")

            children.extend(childs)
        
        self._writeDotty_rec(outputFile, children, i-1)

    '''
    def __str__(self):
        output = cStringIO.StringIO()
        self.writeDotty(output)
        return output.getvalue()
    '''


    # check if the tree satisfies all invariants
    def _check_invariants(self):
        return self._check_nesting() and \
            self._check_covering_tree() and \
            self._check_seperation()


    # check if my_invariant is satisfied:
    # C_i denotes the set of nodes at level i
    # for all i, my_invariant(C_i, C_{i-1})
    def _check_my_invariant(self, my_invariant):
        C = [self.root]
        for i in reversed(range(self.minlevel, self.maxlevel + 1)):
            C_next = sum([p.getChildren(i) for p in C], [])
            if not my_invariant(C, C_next, i):
                print("At level", i, "the invariant", my_invariant, "is false")
                return False
            C = C_next
        return True
        
    
    # check if the invariant nesting is satisfied:
    # C_i is a subset of C_{i-1}
    def _nesting(self, C, C_next, _):
        return set(C) <= set(C_next)

    def _check_nesting(self):
        return self._check_my_invariant(self._nesting)
        
    
    # check if the invariant covering tree is satisfied
    # for all p in C_{i-1} there exists a q in C_i so that
    # d(p, q) <= base^i and exactly one such q is a parent of p
    def _covering_tree(self, C, C_next, i):
        return all(unique(self.distance(p.data, q.data) <= self.base**i
                          and p in q.getChildren(i)
                          for q in C)
                   for p in C_next)

    def _check_covering_tree(self):
        return self._check_my_invariant(self._covering_tree)

    # check if the invariant seperation is satisfied
    # for all p, q in C_i, d(p, q) > base^i
    def _seperation(self, C, _, i):
        return all(self.distance(p.data, q.data) > self.base**i
                   for p, q in product(C, C) if p != q)

    def _check_seperation(self):
        return self._check_my_invariant(self._seperation)
