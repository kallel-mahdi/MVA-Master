class Memory(object):
    def __init__(self):
        
        self.actions = []
        self.rewards = []
        self.masks = []
        self.values = []
        self.entropies = []
        self.log_probs = []
    
    def push(self, action,reward,mask,value,log_prob,entropy):
        
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
    
    def pop_all(self):
        
        actions   = self.actions
        rewards   = self.rewards
        masks     = self.masks
        values    = self.values
        log_probs = self.log_probs
        entropies = self.entropies
        
        self.actions,self.rewards,self.masks, self.values,self.log_probs,self.entropies = [], [], [], [], [], []
        
        
        return  actions,rewards,masks, values,log_probs,entropies