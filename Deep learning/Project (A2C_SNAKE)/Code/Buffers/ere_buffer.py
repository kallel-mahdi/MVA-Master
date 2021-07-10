import random
import numpy as np
from collections import deque

class EREBuffer:

    def __init__(self, max_size,n_updates):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.k = 0
        self.n_updates = n_updates
        self.c_k_min = 5000
        self.c_k = None
        self.mu = 0.996
        

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    
    def sample(self, batch_size , weight = 1):
        
        
        self.k +=1
        c_k =  max (self.max_size * ( (self.mu) ** (1000 * self.k / self.n_updates )),self.c_k_min  )
        c_k = int(c_k)
        self.c_k = c_k
        
        if c_k+100 < len(self.buffer):
            samples = list(self.buffer)[-c_k:]
        else : 
            samples = self.buffer
        
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
    
        batch = random.sample(samples, batch_size)
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)


                


    def __len__(self):
        return len(self.buffer)
    
    
