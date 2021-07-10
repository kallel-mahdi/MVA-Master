import torch 
import numpy as np
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def kernel_product(x, y, mode = "gaussian", gamma = 0.5,add_noise=True,normalize=True,noise=1e-6):

    
    """ computes nxm  pairwise difference matrix"""
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    xmy = pairwise_distances(x,y)
    
    if   mode == "gaussian" : f = lambda  xmy : torch.exp( - gamma * ( xmy ) )
    elif mode == "laplace"  : f = lambda  xmy : torch.exp( - torch.sqrt( xmy + (gamma**2)))
    elif mode == "energy"   : f = lambda  xmy : torch.pow(   xmy + (gamma**2), -.25 )

    ###Sometimes K is not invertible, adding noise to the diagonal helps the problem
    Kxy = f(xmy)
    Kxx = f((x-x).sum(-1))
    Kyy = f((y-y).sum(-1))
    
    if add_noise : Kxy += noise * torch.eye(Kxy.shape[0])
    if normalize : 
      """ Kxy = Kxy / sqrt(Kxx) * sqrt(Kyy)"""
    
      Dxx = torch.diag(torch.pow(Kxx,-0.5))
      Dyy = torch.diag(torch.pow(Kyy,-0.5))

      K = Dxx @ Kxy @ Dyy

    return K.cpu().numpy()


def rbf_kernel(X,sigma):
  return kernel_product(X,X,mode="gaussian",gamma=sigma)