"""
Deep Learning on Graphs - ALTEGRAD - Jan 2021
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn

def normalize_adjacency(A):
    ############## Task 1
    n = A.shape[0]
    A = A + sp.identity(n)
    degs = A @ np.ones(n)
    inv_degs = np.power(degs,-1)
    D_inv = sp.diags(inv_degs)
    A_normalized = D_inv @ A


    ##################
    # your code here #
    ##################

    return A_normalized


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()

    ############## Task 3##################
    
    y_pred,y = list(),list()
    
    idx = adj._indices()
    product = torch.mul(z[idx[0,:],:], z[idx[1,:],:]) ### pointwise product
    y_pred.append(torch.sum(product,dim=1))
    y.append(torch.ones(idx.size(1)).to(device))
    
    idx_r = torch.randint(z.size(0),idx.size())
    product = torch.mul(z[idx_r[0,:],:], z[idx_r[1,:],:]) ### pointwise product
    y_pred.append(torch.sum(product,dim=1))
    y.append(torch.zeros(idx.size(1)).to(device))
    
    y_pred = torch.cat(y_pred,dim=0)
    y = torch.cat(y,dim=0)
    
    ##################
    
    loss = mse_loss(y_pred, y)
    return loss