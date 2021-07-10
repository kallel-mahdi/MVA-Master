import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from operator import mul

class ACN(nn.Module):
    
    def __init__(self,input_dim,output_dim ,use_conv = True , hidden_size =128 ,double_train = False):
        
        super(ACN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_conv = use_conv
        self.double_train = double_train # train jointly the embedding and the AC

        if use_conv and not double_train:
            self.embedding = nn.Sequential(
                nn.Conv2d(self.input_dim[0],32, kernel_size=8, stride=4),
                nn.ReLU(),
                #nn.MaxPool2d(2,2),
                nn.Conv2d(32,64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64,128, kernel_size=3, stride=1),
                nn.ReLU(),
                #nn.MaxPool2d(2,2),
            )        

            self.critic =self.critic =  nn.Sequential(
                nn.Linear(self._get_emb_out(input_dim),hidden_size),
                nn.Linear(hidden_size,1))
        
            self.actor =nn.Sequential(
                nn.Linear(self._get_emb_out(input_dim),hidden_size),
                nn.Linear(hidden_size,self.output_dim))
        
        elif not use_conv and not double_train :
            self.embedding = nn.Sequential(
               nn.Linear(self._prod(input_dim),hidden_size),
               nn.ReLU(),
               nn.Linear(hidden_size,hidden_size),
               nn.ReLU(),
               )        
            self.critic =  nn.Sequential(
                nn.Linear(hidden_size,1))


            
            self.actor = nn.Sequential(
                nn.Linear(hidden_size,self.output_dim))

        elif use_conv and double_train :
            self.embedding = nn.Sequential(
                nn.Conv2d(self.input_dim[0],32, kernel_size=4, stride=2),
                nn.ReLU(),
                #nn.MaxPool2d(2,2),
                nn.Conv2d(32,64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64,128, kernel_size=3, stride=1),
                nn.ReLU(),
                #nn.MaxPool2d(2,2),
            )      

            self.fc =   nn.Linear(self._get_emb_out(input_dim),11)

            self.critic =  nn.Sequential(
                nn.Linear(11,1))
        
            self.actor =nn.Sequential(
                nn.Linear(11,self.output_dim))

        else:
            self.embedding = nn.Sequential(
               nn.Linear(self._prod(input_dim),hidden_size),
               nn.Linear(hidden_size,hidden_size),
               nn.Linear(hidden_size,hidden_size),
            )   

            self.fc = nn.Linear(hidden_size,11)
            self.critic =  nn.Sequential(
                nn.Linear(11,1))


            
            self.actor = nn.Sequential(
                nn.Linear(11,self.output_dim))



        
    
        
        
    
    def forward(self,x):
        x = x.unsqueeze(0)
        if not self.use_conv :
            x = torch.reshape(x , (x.size(0),-1)) 
        x = self.embedding(x)
        if self.double_train:
            x = x.view(x.size(0),-1)
            y = self.fc(x)
            value = self.critic(y.clone())
            
            policy = Categorical(logits = self.actor(y.clone()))
            return value,policy , y

        else:
            x = x.view(x.size(0),-1)
            value = self.critic(x)
            
            policy = Categorical(logits = self.actor(x))
            return value,policy 

    

    def _get_emb_out(self, shape):
         o = self.embedding(torch.zeros(1, *shape))
         return int(np.prod(o.size()))
    def _prod(self,lst):
        p = 1
        for e in lst:
            p *=e
        return p

        
