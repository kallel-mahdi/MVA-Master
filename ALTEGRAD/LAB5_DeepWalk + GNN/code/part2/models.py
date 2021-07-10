"""
Deep Learning on Graphs - ALTEGRAD - Dec 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        x1 = torch.mm(x_in,self.fc1.weight.t())
        x1 = torch.mm(adj,x1)
        x1 = self.dropout(self.relu(x1))
        
        x2 = torch.mm(x1,self.fc2.weight.t())
        x2 = torch.mm(adj,x2)
        x2_drop = self.dropout(self.relu(x2))
        
        x =  torch.mm(x2_drop,self.fc3.weight.t())
        ##################

        return F.log_softmax(x, dim=1), x2.detach().cpu().numpy()