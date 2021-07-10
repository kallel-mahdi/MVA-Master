import torch
# import gym
import sys, os
sys.path.append('../')
import numpy as np
from Envs.environment_snake import *
import matplotlib.pyplot as plt
from Agents.dqn_agent import *
from Agents.ac_agent import *
import os
os.environ['SDL_VIDEODRIVER']='dummy'
pygame.init()


def acn_experience(agent_n='ACN',use_conv=False,double_train=False,n_steps=20,bootstrap=False,
                   learning_rate=1e-4,img=False,train_steps = 1000000,eval_every=1000,weighted_sampling=1,clip_grad=True):
    
    env = SnakeGame(w=80, h=80 , max_reward = 1,food_nb = 1 , early_stopping_factor = 10 ,gray_scale = False,
                    isdisplayed = False, use_images = img   , image_reduction = 2 )
    
    if agent_n== 'ACN':
        
        agent = ACAgent(env, use_conv=use_conv, double_train= double_train,\
            learning_rate=learning_rate,n_steps=n_steps,bootstrap=bootstrap,clip_grad=clip_grad)
        agent.train(n_updates = train_steps,eval_every=eval_every)
        agent.plot()
        
if __name__=='__main__':
    acn_experience()
