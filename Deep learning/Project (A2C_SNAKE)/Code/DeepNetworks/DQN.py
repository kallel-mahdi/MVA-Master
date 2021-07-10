
import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from operator import mul
class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim , use_conv = False , dueling= False , hidden_size = 128):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dueling = dueling
        self.relu = nn.ReLU()
        if use_conv :
            self.embedding = nn.Sequential(
                nn.Conv2d(self.input_dim[0],16, kernel_size=8, stride=4),
                nn.ReLU(),
                #nn.MaxPool2d(2,2),
                nn.Conv2d(16,32, kernel_size=4, stride=2),
                nn.ReLU(),
                #nn.MaxPool2d(2,2),
            )      

            self.fc = nn.Sequential(
            nn.Linear(self._get_emb_out(input_dim), self.output_dim)) 

            self.fc2_A = nn.Linear(self._get_emb_out(input_dim), hidden_size)
            self.fc3_A = nn.Linear(hidden_size, self.output_dim)
            
            self.fc2_V = nn.Linear(self._get_emb_out(input_dim), hidden_size)
            self.fc3_V = nn.Linear(hidden_size, 1)
        
        else :
            self.embedding = nn.Sequential(
               nn.Linear(self._prod(input_dim),hidden_size),
               nn.ReLU(),
               nn.Linear(hidden_size,hidden_size),
               nn.ReLU(),
            )        
            self.fc = nn.Sequential(
            nn.Linear(hidden_size, self.output_dim))

            self.fc2_A = nn.Linear(hidden_size, hidden_size)
            self.fc3_A = nn.Linear(hidden_size, self.output_dim)
            
            self.fc2_V = nn.Linear(hidden_size, hidden_size)
            self.fc3_V = nn.Linear(hidden_size, 1)  

        

    def forward(self, state):
        features = self.embedding(state)
        features = features.view(features.size(0), -1)

        if self.dueling:
            x_A = self.relu(self.fc2_A(features))
            A = self.fc3_A(x_A)
            x_V = self.relu(self.fc2_V(features))
            V = self.fc3_V((x_V))
            Q = V + A - torch.mean(A,dim=1).unsqueeze(1)
        else : 
            Q = self.fc(features)
        
        return Q

    def _get_emb_out(self, shape):
         o = self.embedding(torch.zeros(1, *shape))
         return int(np.prod(o.size()))
    def _prod(self,lst):
        p = 1
        for e in lst:
            p *=e
        return p

