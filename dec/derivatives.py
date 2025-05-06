import numpy as np
import scipy.sparse as sp

def build_d0(points, edges):
    rows, cols, data = [], [], []
    for i, (a, b) in enumerate(edges):
        rows += [i, i]
        cols += [a, b]
        data += [-1, +1]
    return sp.coo_matrix((data, (rows, cols)), shape=(len(edges), len(points))).tocsr()

def build_d1(triangles, edge_map):
    rows, cols, data = [], [], []
    for t_idx, tri in enumerate(edge_map):
        for edge_idx, sign in tri:
            rows.append(t_idx)
            cols.append(edge_idx)
            data.append(sign)
    return sp.coo_matrix((data, (rows, cols)), shape=(len(triangles), max(cols)+1)).tocsr()