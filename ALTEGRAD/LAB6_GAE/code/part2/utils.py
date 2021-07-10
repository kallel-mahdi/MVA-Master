"""
Graph Mining - ALTEGRAD - Jan 2021
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset():
    Gs = list()
    y = list()

    p1 = 0.2
    p2 = 0.4
    for i in range(50):
        n = randint(10,20)
        Gs.append(nx.fast_gnp_random_graph(n,p=p1))
        y.append(0)
        
        n = randint(10,20)
        Gs.append(nx.fast_gnp_random_graph(n,p=p2))
        y.append(1)
                                       
    ############## Task 6
    
    ##################
    # your code here #
    ##################

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
