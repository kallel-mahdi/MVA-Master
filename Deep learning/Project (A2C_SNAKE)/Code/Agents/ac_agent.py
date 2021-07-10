# %% 
import torch
import sys
sys.path.append('../')
from DeepNetworks.ACN import ACN
from Buffers.memory import Memory
from collections import deque
import numpy as np
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn

class ACAgent:
    
    def __init__(self,env, use_conv=True ,double_train=False, gamma = 0.95 , learning_rate= 1e-4,
                 n_steps=20,bootstrap=True,clip_grad=False):

        self.model = ACN(env.observation_space, len(env.actions),use_conv= use_conv,double_train=double_train)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.memory = Memory()
        self.gamma = gamma
        self.n_steps = n_steps
        self.bootstrap = bootstrap
        self.optimizer =  torch.optim.Adam(self.model.parameters() , lr = learning_rate)
        self.env = env
        self.use_conv = use_conv
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad
        
        self.name = "IMG_"*self.use_conv +'ACN'+  double_train * '_DOUBLE' +'_BS'*self.bootstrap + \
            '_NS_'+ str(self.n_steps)+ "_LR_"+str(self.learning_rate)+ '_GAMMA_' + str(self.gamma) + '_CLIP_'*self.clip_grad
     
        
    
    
    def select_action(self,state ,double_train =True): 
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        if double_train:
            value,policy,_ = self.model(state_tensor)
        else:
            value,policy= self.model(state_tensor)

        action = policy.sample()
        
        return action
    
        
   
    
    def compute_returns(self,last_value, rewards, masks , bootstrap=False): 
        
        r = last_value

        if not bootstrap :
            r = 0
        returns = []
        for step in reversed(range(len(rewards))):
            r = rewards[step] + self.gamma * r * masks[step]
            returns.insert(0, r)
            
        return returns
    
    
    def compute_loss(self,next_state,actions,rewards,masks,values,log_probs,entropies,embedding_loss= 0,bootstrap = False):
        
        
        next_state = torch.FloatTensor(next_state).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        masks = torch.cat(masks).to(self.device)
        values = torch.cat(values).to(self.device)
        log_probs = torch.cat(log_probs).to(self.device)
        entropies = torch.cat(entropies).to(self.device)
        
        with torch.no_grad():
            
            if bootstrap:
                if embedding_loss !=0:
                    next_values,_ , _ = self.model(next_state)
                else:
                    next_values,_  = self.model(next_state)
            else :
                next_values = np.zeros(1)
                
            returns = self.compute_returns(next_values.squeeze(), rewards, masks,bootstrap)
            returns = torch.FloatTensor(returns).to(self.device)
        
        
        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        #entropy_loss = entropies.mean()

        #loss = 1 * actor_loss + 1 * critic_loss +1* embedding_loss#+ 0.001 * entropy_loss
               
               
        return actor_loss, critic_loss,embedding_loss
      
        def eval(self,double_train= True,n_sim=5,display = False):
          """
          Monte Carlo evaluation of DQN agent
          """
          #copy_env = self.env.copy()
          self.env.set_display(display)
          #copy_env = self.env
          rewards = np.zeros(n_sim)
          scores = np.zeros(n_sim)
          for sim in range(n_sim):
              state = self.env.reset()
              done = False
              while not done:

                  action = self.select_action(state , double_train)
                  next_state, reward, done, score = self.env.step((action.unsqueeze(1).item()))
                  rewards[sim] += reward
                  scores[sim] = score
                  state = next_state
          return rewards ,  scores

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))

    
    def train(self,n_updates=10000, eval_every = 1000,n_sim =100 , double_train = False):
        
        next_state = self.env.reset()
        episodes_rewards =[]
        start = time.time()
        episode = 0
        losses = []
        
        for update in range(1,n_updates):
            y = None
            target = None
            embedding_loss = 0
     
            if self.bootstrap :
                step=0
                # if int(np.log(update))%10000==0:
                #     self.n_steps=+1
                done = False


                while step <self.n_steps and not done:
                    step+=1

                    next_state_tensor = torch.FloatTensor(next_state).to(self.device)

                    if double_train:
                        value, policy , y = self.model.forward(next_state_tensor)
                    else:
                        value, policy = self.model.forward(next_state_tensor)

                    action = policy.sample()
                    #print(action.unsqueeze(1).item())
                    next_state, reward, done, _  = self.env.step(action.unsqueeze(1).item()) 

                    if double_train: 
                        target = self.env.get_discreet()
                        target = torch.FloatTensor(target).to(self.device).unsqueeze(0)
                        embedding_loss += nn.BCEWithLogitsLoss()(y , target)


                    #mask = torch.from_numpy(1.0 - done)
                    mask = torch.tensor([1-done])
                    ########################
                    log_prob = policy.log_prob(action)
                    entropy  = policy.entropy()
                    ########################

                    self.memory.push(action,reward,mask,value,log_prob,entropy)

            else : 
                done = False
                while not done :
                    
                    next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                    if double_train:
                        value, policy , y = self.model.forward(next_state_tensor)
                    else:
                        value, policy = self.model.forward(next_state_tensor)

                        

                    ########################
                    action = policy.sample()
                    #print(action.unsqueeze(1).item())
                    next_state, reward, done, _  = self.env.step(action.unsqueeze(1).item())   
                    
                    if double_train: 
                        target = self.env.get_discreet()
                        target = torch.FloatTensor(target).to(self.device).unsqueeze(0)
                        embedding_loss += nn.BCEWithLogitsLoss()(y , target)

                    #mask = torch.from_numpy(1.0 - done)
                    mask = torch.tensor([1-done])
                    ########################
                    log_prob = policy.log_prob(action)
                    entropy  = policy.entropy()
                    ########################
                        

                    self.memory.push(action,reward,mask,value,log_prob,entropy)



            data = self.memory.pop_all()
            if done:
                actor_loss, critic_loss,embedding_loss = self.compute_loss(next_state,*data , embedding_loss = embedding_loss, bootstrap=False)
            else:
                actor_loss, critic_loss,embedding_loss = self.compute_loss(next_state,*data ,embedding_loss = embedding_loss, bootstrap=True)
            

            loss = actor_loss+critic_loss+embedding_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) ## Clip gradients to 0.5
            self.optimizer.step()
            
            if done : 
                
                next_state = self.env.reset()
                episode +=1
            
            if update % eval_every == 0:

                print(actor_loss, critic_loss,embedding_loss)



                end = time.time()
                #total_num_steps = (update + 1) * args.num_envs * args.num_steps
                total_num_steps = (update + 1) * self.n_steps
                rewards , scores = self.eval(double_train,n_sim=n_sim)
                std_rewards = np.std(rewards)
                mean_rewards = np.mean(rewards)
                std_scores = np.std(scores)
                mean_scores = np.mean(scores)
                t1 = end-start
                episodes_rewards.append([update,t1,mean_rewards,std_rewards ,mean_scores,std_scores])
                print("********************************************************")
                print("update: {0}, total steps: {1}, FPS: {2}".format(update, total_num_steps, int(total_num_steps / (end - start))))
                print("episode =", episode, ", reward = ", np.round(mean_rewards,2) , "scores = ", np.round(mean_scores,2))
                print("********************************************************")
                torch.save(self.model.state_dict(), f'Experiments/model_{type(self.model).__name__}_{type(self.env).__name__}_{self.use_conv}_{random.randint(1,10)}_descreet.pth')
        np.savetxt(f"Experiments/rewards_{self.name}.txt", np.array(episodes_rewards), fmt="%s")
        return episodes_rewards
    
    def eval(self,double_train= True,n_sim=5,display = False):
    
        """
        Monte Carlo evaluation of DQN agent
        """
        #copy_env = self.env.copy()
        self.env.set_display(display)
        rewards = np.zeros(n_sim)
        scores = np.zeros(n_sim)
        for sim in range(n_sim):
            state = self.env.reset()
            done = False
            while not done:

                action = self.select_action(state , double_train)
                next_state, reward, done, score = self.env.step((action.unsqueeze(1).item()))
                rewards[sim] += reward
                scores[sim] = score
                state = next_state
        return rewards ,  scores

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))  
    def plot(self, reward = True , score = True):
        
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
            plt.plot(exp[:,0], exp[:,4],label = 'Using embedding loss')
            plt.fill_between(exp[:,0], exp[:,4]+exp[:,5] , np.clip(exp[:,4]-exp[:,5],0,None) , alpha = 0.2)
#             plt.plot(exp2[:,0], exp2[:,4], label = 'Without embedding loss')
#             plt.fill_between(exp2[:,0], exp2[:,4]+exp2[:,5] , np.clip(exp2[:,4]-exp2[:,5],0,None) , alpha = 0.2 )
#             plt.plot(exp3[:,0], exp3[:,4] , label = 'Using discreet representation')
#             plt.fill_between(exp3[:,0], exp3[:,4]+exp3[:,5] , np.clip(exp3[:,4]-exp3[:,5],0,None) , alpha = 0.2)
            plt.xlabel('training steps')
            plt.legend()
        plt.show()