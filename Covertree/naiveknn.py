# File: naiveknn.py
# Author: Nil Geisweiller
# email: ngeiswei@gmail.com
# Date: 
# All Rights Reserved. Copyright Nil Geisweiller 2011
#
# This function to perform maive nearest neighbors.

from heapq import nsmallest 

def knn(k, pt, pts, dist_f):
    '''Return the k-nearest points in pts to pt using a naive
    algorithm. dist_f is a function that computes the distance between
    2 points'''
    return nsmallest(k, pts, lambda x: dist_f(x, pt))

def nn(pt, pts, dist_f):
    '''Like knn but return the nearest point in pts'''
    return knn(1, pt, pts, dist_f)[0]
