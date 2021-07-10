#%%
if __name__ == "__main__":
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




    agent_name = 'ACN'
    Train_agent = False
    plot_results = False

    display = True
    Image_representation = True
    BATCH_SIZE = 256
    TRAIN_STEPS = 100000
    EVAL_EVERY = 100
    WEIGHTED_SAMPLING = 1
    double_train = True #  using embedding loss
    double = False
    dueling = False
    use_conv = Image_representation

    # # define environement

    env = SnakeGame(w=80, h=80 , max_reward = 1,food_nb = 1 , early_stopping_factor = 10 ,gray_scale = False, isdisplayed = display, use_images = Image_representation   , image_reduction = 2 )
    if agent_name == 'DQN':
        agent = DQNAgent(env, use_conv=use_conv,dueling=dueling, double = double,batch_size = BATCH_SIZE)
    if agent_name == 'ACN':
        agent = ACAgent(env , use_conv=use_conv ,double_train=double_train)
    if Train_agent:
        if agent_name == 'DQN':
            agent.train(TRAIN_STEPS,EVAL_EVERY)
        if agent_name == 'ACN':
            agent.train(n_updates=TRAIN_STEPS,n_sim =100 , eval_every =EVAL_EVERY , double_train = double_train)

    if plot_results:
        agent.plot()

    if display:
        agent.load_model('Experiments\model_ACN_SnakeGame_Best.pth')
        if agent_name == 'ACN':
            agent.eval(double_train=True,display = True,n_sim=10)
        if agent_name == 'DQN':
            agent.eval(display = True,n_sim=10)









# %%
