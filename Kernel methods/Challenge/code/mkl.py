import numpy as np
from classifiers import KSVM

class MKL():

  def __init__(self,kernels,y_train):

    self.c = 1e-5
    self.kernels = kernels
    self.m = len(kernels)
    self.X_train = X_train.cpu().numpy()
    self.y_train = y_train
    self.diag_y = np.diag(y_train)*1.
    self.first_iteration = True
    self.svm = KSVM(self.c)

  
  def fix_weight_precision(self,d,weight_precision):
    new_d=d.copy()
    # zero out weights below threshold
    new_d[np.where(d<weight_precision)[0]]=0
    # normalize
    new_d=new_d/np.sum(new_d)
    return new_d
  
  def grads(self,alphas):

    dJ = np.zeros(self.m)

    for i in range(self.m):
      dJ[i] = -0.5*  alphas.T @ self.diag_y @ self.kernels[i] @  self.diag_y @ alphas
    
    print('_______DJ_____:',dJ)
    return dJ

  def compute_descent_direction(self,d, dJ,mu):
    
    D = np.zeros_like(dJ)
    D[(d<1e-5) & (dJ-dJ[mu]>1e-6) ]=0
    D[d>1e-5] = - dJ[d>1e-5] + dJ[mu] ## D_mu = 0 :DD
    D[mu] = np.sum(dJ[dJ>1e-6] -dJ[mu])
    
    return D
    

  def stopping_criterion(self,dJ, d, eps):
    M = len(dJ)
    if self.first_iteration:
        self.first_iteration = False
        return False

    else:
        dJ_plus = dJ[d>1e-10]
        print("dJ_plus",dJ_plus)

        dJ_min = np.min(dJ_plus)
        dJ_max = np.max(dJ_plus)

        dJ_zero= dJ[(d<1e-5)] 

        if len(dJ_zero)==0:
          print("breaking")
          return True

        print("dJ__zero",dJ_zero)
        dJ_min_zero = np.min(dJ_zero)

        return ((dJ_max - dJ_min) <= eps) and dJ_min_zero >= dJ_max


  def compute_kernel(self,d):

    K = np.zeros_like(self.kernels[0])
    for i in range(self.m): K += d[i] * self.kernels[i]
    return K
  
  
  def compute_J(self,K_train):

      
      self.svm.fit(self.X_train,self.y_train,K_train)
      alphas = self.svm.alphas
      J = self.svm.solution_value

      return alphas,J,_


  def backtrack(self,gamma0,d,Jd,D,dJ,c=0.5,T=0.5):

    
    #m = D' * dJ, should be positive
    #Loop until f(x + gamma * p) >= f(x) + gamma*c*m
    # J(d + gamma * D) >= J(d) + gamma * c * m
    gamma = gamma0
    m = D.T @ dJ
    print("_____M__:",m)

    print("backtracking")
    while True:
        
        new_d = d + gamma * D
        K = self.compute_kernel(new_d)

        alpha, new_J, info = self.compute_J(K)

        print("gamma",gamma,"new_D",new_d,"new_J",new_J,"objective",Jd + gamma * c * m)

        if new_J >= Jd + gamma * c * m or gamma < .1:
            return gamma
        else:
            #Update gamma
            gamma = gamma * T

    return gamma / 2
  
  def fit(self):

    d = np.ones(self.m)/self.m
    dJ = -20 * np.ones(self.m) ## just to start the loop


    while not self.stopping_criterion(dJ,d,0.01):

      K = self.compute_kernel(d)
      alphas,J,_ = self.compute_J(K)
      dJ = self.grads(alphas)
      mu = np.argmax(d)
      D = self.compute_descent_direction(d,dJ,mu)
      J_cross = 0
      d_cross = d.copy()
      D_cross = D.copy()
      
    
      #Get maximum admissible step size in direction D
      sub_iteration = 0
      while (J_cross < J):
          if sub_iteration !=0: J = J_cross.copy()
          
          sub_iteration += 1
          d = d_cross.copy()
          D = D_cross.copy()

          #Find gamma_max and v
          D_neg = D[D<0]
          d_neg = d[D<0]
          v = np.argmin(-d_neg /D_neg)
          gamma_max = (-d_neg/D_neg)[v]

          ## update descent direction
          d_cross = d + gamma_max * D
          D_cross[mu] = D[mu] - D[v]
          D_cross[v] = 0

          ### compute new objective value

          K_cross = self.compute_kernel(d_cross)
          alpha_cross, J_cross, cross_info = self.compute_J(K_cross)
          print ('new J cross', J_cross)

      #Line search along D for gamma (step) in [0, gamma_max]
      # gamma = helpers.get_armijos_step_size()
      gamma = self.backtrack(gamma_max,d,J_cross,D, dJ)
      print('gamma:', gamma)
      print ('D:', D)
      d += gamma * D
      d = self.fix_weight_precision(d, 1e-6)

    #Return final weights
    return d

if __name__ == '__main__':
    
    K2=kernel_product(X_train,X_train,mode='gaussian',add_noise=True).cpu().numpy()
    K1=kernel_product(X_train,X_train,mode='energy',add_noise=True).cpu().numpy()
    kernels_list = [K1,K2]
    mkl = MKL(kernels_list,y_train)
    d= mkl.fit()
    print(d)