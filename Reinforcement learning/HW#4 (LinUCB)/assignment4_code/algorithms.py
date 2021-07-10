import numpy as np 

import math
from math import log, sqrt
from numpy.linalg import inv

class LinUCB:

    def __init__(self, 
        representation,
        reg_val, noise_std, delta=0.01 ):
        
        self.representation = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.param_bound = representation.param_bound
        self.features_bound = representation.features_bound
        self.delta = delta
        self.nd = self.representation.nd
        self.b_max = 0
        self.reset()
        self.maxa_bound = None
        

    def reset(self):
        ### TODO: initialize necessary info
        self.A_inv = (1/self.reg_val) * np.eye(self.nd)
        self.b = 0
        self.theta =   np.zeros(self.nd)
        ###################################
        self.t = 1

    def sample_action(self, context):
        ### TODO: implement action selection strategy
        
        L = self.features_bound
        S = self.param_bound
        lamda = self.reg_val
        sigma = self.noise_std
        delta = self.delta
        d = self.nd
        #############
        phi = self.representation.get_features(context)
        act_reprs = self.representation.get_features(context) ### returns representations phi(u,a) for all A
        act_means = (act_reprs @ self.theta).reshape(-1)
        act_intvs = np.sqrt(np.diag( act_reprs@ self.A_inv @ act_reprs.T)) ## the diagonal elements of this matrix are our entries.
        alpha_t = sigma * sqrt(d*log( (1 + (self.t * L /lamda))/delta)) + sqrt(lamda )* S
        act_bounds = act_means + alpha_t * act_intvs    
        maxa = np.argmax(act_bounds)
        self.maxa_bound = 2 * alpha_t * act_intvs[maxa]
        ###################################
        if alpha_t > self.b_max : self.b_max = alpha_t
        self.t += 1
        return maxa

    def update(self, context, action, reward):
        phi = self.representation.get_features(context, action).reshape(-1,1)
        ### TODO: update internal info (return nothing)
        self. b += phi * reward
        self.A_inv -= (self.A_inv@ phi @ phi.T @ self.A_inv) / (1 + phi.T @ self.A_inv @ phi)
        self.theta = self.A_inv @ self.b

        ###################################


class RegretBalancingElim:
    def __init__(self, 
        representations,
        reg_val, noise_std,delta=0.01):
        
        self.representations = representations
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.delta = delta
        #self.c = 1
        self.c = 1e-3
        self.M = len(representations)
        self.last_selected_rep = None
        self.active_reps = None # list of active (non-eliminated) representations
        self.t = None
        
        self.reset()
    

    def reset(self):
        ### TODO: initialize necessary info
        self.learners = [LinUCB(rep,self.reg_val,self.noise_std,self.delta) for rep in self.representations]
        ## We need to run each learner 2 times to avoid nan in log.
        self.lstart = list(range(self.M)) ## list of learners to start 
        
        self.active_reps = [str(i) for i in range(self.M)]
        self.u_i = [0 for learner in self.learners] ### rewards
        self.n_i = [0 for learner in self.learners]
        self.r_i = [0 for learner in self.learners] ### regrets
        
    
        ###################################
        self.t = 1
    
    def optimistic_action(self, rep_idx, context):
        ### TODO: implement action selection strategy given the selected representation
        maxa = self.learners[rep_idx].sample_action(context)
        ###################################
        return maxa

    def sample_action(self, context):
        ### TODO: implement representation selection strategy
        #         and action selection strategy
        
        if self.t <= 2* self.M:
            ## warmup each leearner 2 times.
            
            self.last_selected_rep = self.lstart[(self.t-1) // 2]
        else :
            self.last_selected_rep = np.argmin(self.r_i)
            
        action = self.optimistic_action(self.last_selected_rep,context)

        ###################################
        self.t += 1
        return action
    
    def delete(somelist,idxs):
    
        lst  = [i for j, i in enumerate(somelist) if j not in idxs]
        return lst

  
    
    
    def deactivate_learners(self,idxs):
            
        def delete(somelist,idxs):
            lst  = [i for j, i in enumerate(somelist) if j not in idxs]
            return lst
        
        self.u_i = delete(self.u_i,idxs)
        self.n_i = delete(self.n_i,idxs)
        self.r_i = delete(self.r_i,idxs)
        self.active_reps = delete(self.active_reps,idxs)
        self.learners = delete(self.learners,idxs)
    
    

    def update(self, context, action, reward):
        
        idx = self.last_selected_rep
        learner = self.learners[idx]
        learner.update(context,action,reward)
        ### TODO: implement update of internal info and active set 
        L = learner.features_bound
        S = learner.param_bound
        lamda = learner.reg_val
        sigma = learner.noise_std
        delta = learner.delta
        d = learner.nd
        t = learner.t
        b_max = learner.b_max
        maxa_bound = learner.maxa_bound
        
        #self.r_i[idx] = 2*b_max * sqrt(d * t * (1 +  L**2/lamda) * log ((d * lamda + t * L**2)/ (d* lamda)) )
        self.r_i[idx] += maxa_bound
        self.u_i[idx] += reward
        self.n_i[idx] += 1
        
    
        
        # if self.t == 2*self.M+1:
        #     print("Self n_i",self.n_i)
        
        if self.t > 2*self.M :
        
        
            u_i = np.array(self.u_i).reshape(-1)
            r_i = np.array(self.r_i).reshape(-1)
            n_i = np.array(self.n_i).reshape(-1)
            lower_bounds = u_i / n_i - self.c * np.sqrt( np.log(( self.M * np.log(n_i))/delta)/n_i)
            upper_bounds = (u_i + r_i) /n_i + self.c * np.sqrt( np.log(( self.M * np.log(n_i))/delta)/n_i)
            butcher = np.max(lower_bounds)
            
            
            to_elim_o = np.where(upper_bounds <= butcher)
            to_elim = to_elim_o[0].tolist()
            
            ## delete aftrwards
            if len (to_elim) >= 1 and len(self.learners)>1:
                #print("To elim",to_elim)
                self.deactivate_learners([to_elim[0]])
                
                
            
        ###################################




