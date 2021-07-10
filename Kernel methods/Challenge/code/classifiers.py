import cvxpy as cp
import numpy as np
import torch 


n = 1800

class KRR():

  def __init__(self,lamda):
    
    self.lamda = torch.FloatTensor([lamda])

  def fit(self,K_train,y_train):

    K_train = torch.FloatTensor(K_train)
    n = torch.FloatTensor(K_train.shape[0])
    eye = torch.eye(K_train.shape[0])
    
    A = K_train + self.lamda*n*eye
    self.alphas = torch.linalg.solve(A,y_train) ### (K + lda * n * I) **-1 @ y
  
  def predict(self,K_test):
    K_test = torch.FloatTensor(K_test)
    return  K_test @ self.alphas

class KSVM():

  def __init__(self, C=0.1e-9,train_size=n):

    self.C = C
    self.train_size = int(train_size)


    ### don't forget to change n
    self._alpha = cp.Variable((self.train_size, 1))
    self._y  = cp.Parameter((self.train_size, 1))
    self._K  = cp.Parameter((self.train_size, self.train_size),PSD=True) 

    ### dual objective
    objective =  cp.Maximize(self._alpha.T @ self._y -0.5* cp.quad_form(self._alpha,self._K))  ## slide 212/914
    constraints = [cp.multiply(self._alpha,self._y)  >=0,cp.multiply(self._alpha,self._y)  <=self.C]
    self.problem = cp.Problem(objective, constraints)
  

  
  def fit(self,K_train,y_train):
  

    diag_y = np.diag(y_train)*1. ## to float
    self._y.value = y_train.reshape(-1,1)
    self._K.value =   K_train
    
    # solve the dual problem
    self.solution_value = self.problem.solve(warm_start=True)
    print("SVM solve time:", self.problem.solver_stats.solve_time)
    
    # alpha is the solution of the primal problem
    self.alphas = self._alpha.value
  
  def predict(self,K_test):

    return K_test @ self.alphas
    

