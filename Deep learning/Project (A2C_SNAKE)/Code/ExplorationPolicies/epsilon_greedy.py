import numpy as np
import random

class EpsilonGreedy :
    
    def __init__(self,agent,epsilon=0.05):
        
        self.agent = agent
        #self.n_actions = self.agent.env.action_space.n
        self.n_actions = len(self.agent.env.actions)
        self.epsilon_min = epsilon
        self.epsilon_start = 1
        self.decrease_epsilon = 200
        
        self.epsilon = self.epsilon_start ## initialize
        
    def update(self):
        
        
        if self.epsilon > self.epsilon_min:
             
            self.epsilon -= (self.epsilon_start - self.epsilon_min) / self.decrease_epsilon
        
        
    def select_action(self,state):

        p = random.random()
                        
        if p < self.epsilon:  
        
            action = np.random.randint(0,self.n_actions)
            
        else :
            action = self.agent.select_greedyaction(state)

        return action
                
                