import torch 
import torch.nn as nn
import sys, os
sys.path.append('../')
import numpy as np
from Buffers.basic import BasicBuffer
from Buffers.ere_buffer import EREBuffer
from DeepNetworks.DQN import DQN
from ExplorationPolicies.epsilon_greedy import EpsilonGreedy
import time
from copy import deepcopy
import matplotlib.pyplot as plt

class DQNAgent:

    def __init__(self, env, use_conv=False,dueling=False, double = False,
                 learning_rate=1e-4, gamma=0.99, buffer_size=100000,
                 batch_size = 256,update_tgt_every=200,weighted_sampling=1,ERE = False):
        
        self.env = env
        self.use_conv = use_conv
        
        if ERE:
            self.replay_buffer = EREBuffer(max_size=buffer_size,n_updates = 100000)
        else :
            self.replay_buffer = BasicBuffer(max_size=buffer_size)
            
        self.policy = EpsilonGreedy(self)
        self.dueling = dueling
        self.double = double
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update_tgt_every = update_tgt_every
        self.weighted_sampling = weighted_sampling
        self.ERE = ERE
        self.name = "IMG_"*self.use_conv +'DQN'+ dueling * '_DUELING' + double * '_DOUBLE' + '_BS_'+ str(self.batch_size )+ \
            "_LR_"+str(self.learning_rate)+'_UTE_'+str(self.update_tgt_every)+"_WS_" + str(self.weighted_sampling) + '_ERE'*self.ERE
        self.model = DQN(env.observation_space, len(self.env.actions),use_conv =self.use_conv, dueling=dueling)  
        self.tgt_model = deepcopy(self.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.gamma = gamma  
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr =self.learning_rate )
        self.MSE_loss = nn.MSELoss()
        self.learn_steps = 0 
        
        
        
    
    def select_greedyaction(self, state):
        with torch.no_grad():
            # ====================================================
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            Q = self.model.forward(tensor_state)
            action_index = torch.argmax(Q,1)
            
            # ====================================================
        return action_index.item()
    
    def select_action(self,state):
        
        action = self.policy.select_action(state)
        
        return action
        

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            
            if self.double:

                tgt_Q = self.tgt_model.forward(next_states)
                real_Q = self.model.forward(next_states)

                greedy_Q = torch.max(tgt_Q,dim=1)[0].unsqueeze(1)
                greedy_A = torch.max(tgt_Q,dim=1)[1].unsqueeze(1)
            
                unb_Q    =  real_Q.gather(1,greedy_A).squeeze() 
                expected_Q =  rewards.squeeze(1) + (1 - dones).squeeze() * self.gamma * unb_Q
    
            else :
                     
                tgt_Q = self.tgt_model.forward(next_states)
                max_next_Q = torch.max(tgt_Q, 1)[0]

                expected_Q = rewards.squeeze(1) + (1 - dones).squeeze() * self.gamma * max_next_Q
        

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        #print(expected_Q.size())
        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, WEIGHTED_SAMPLING):

        batch = self.replay_buffer.sample(self.batch_size,WEIGHTED_SAMPLING)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        self.policy.update() ## maybe update(self.replay_buffer[-1]?)
        self.learn_steps +=1
        
        if self.learn_steps % self.update_tgt_every == 0:
            
            self.tgt_model.load_state_dict(self.model.state_dict())
   
    def eval(self, display = False, n_sim=50):
        
        """
        Monte Carlo evaluation of DQN agent
        """
        rewards = np.zeros(n_sim)
        scores = np.zeros(n_sim)
        self.env.set_display(display)
        #copy_env = self.env
        # Loop over number of simulations
        for sim in range(n_sim):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, score = self.env.step(action)
                # update sum of rewards
                rewards[sim] += reward
                scores[sim] = score
                state = next_state
        return rewards , scores
    

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))  
        self.tgt_model.load_state_dict(torch.load(path))  


    def train(self,TRAIN_STEPS,EVAL_EVERY):

        t0 = time.time()
        self.env.set_display(False)
        env = self.env
        #env = self.env
        state = env.reset()
        episodes_rewards =[]
        ep= 0
        episode_reward = 0
        total_time = 0  
        train_step = 1
        
        if self.ERE: self.replay_buffer.n_updates = TRAIN_STEPS
        
        
        while  train_step< TRAIN_STEPS:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_time +=1 
            episode_reward += reward
            transition = (state, action, reward, next_state, done)
            self.replay_buffer.push(*transition)
            
            if len(self.replay_buffer) > self.weighted_sampling*self.batch_size:
                train_step +=1

                if train_step%1000 == 0:
                    print('training step : ',train_step)
                self.update(self.weighted_sampling)

                
            if done  :
                state = env.reset()
                ep += 1
                episode_reward = 0


            
            else : 

                state  = next_state
        
            if train_step % EVAL_EVERY == 0:

                    # evaluate current policy
                    rewards ,scores = self.eval(n_sim=100)
                    mean_rewards = np.mean(rewards)
                    std_rewards = np.std(rewards)

                    mean_scores = np.mean(scores)
                    std_scores = np.std(scores)

                    t1 = time.time()          
                    print("episode =", ep, ", greedy reward = ", np.round(np.mean(rewards),2),",mean score = ",mean_scores , "time = ", t1-t0)
                    
                    if self.ERE: print("Taking last",self.replay_buffer.c_k,"from buffer")
                    episodes_rewards.append([train_step,t1-t0,mean_rewards,std_rewards ,mean_scores,std_scores])
                    

                    torch.save(self.model.state_dict(), f'Experiments/model_{type(self.model).__name__}_{type(self.env).__name__}_{self.name}_.pth')

                    np.savetxt(f"Experiments/rewards_{self.name}.txt", np.array(episodes_rewards), fmt="%s")

        return episodes_rewards



    
    def plot(self , reward = True , score = True):
        
        exp = np.loadtxt(f"Experiments/rewards_{self.name}.txt")
        if reward:
            plt.figure()
            plt.title('Mean reward over learning')
            plt.plot(exp[:,0], exp[:,2])
            plt.fill_between(exp[:,0], exp[:,2]+exp[:,3] , exp[:,2]-exp[:,3] , alpha = 0.2)
            plt.xlabel('training steps')
        if score:
            plt.figure()
            plt.title('Mean score over learning')
            plt.plot(exp[:,0], exp[:,4])
            plt.fill_between(exp[:,0], exp[:,4]+exp[:,5] ,np.clip(exp[:,4]-exp[:,5],0,None), alpha = 0.2)
            plt.xlabel('training steps')
        
