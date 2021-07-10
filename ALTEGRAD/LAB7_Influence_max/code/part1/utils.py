"""
Learning on Sets - ALTEGRAD - Jan 2021
"""

import numpy as np
from random import shuffle

def create_train_dataset():
    n_train = 100000
    max_train_card = 10
    X_train = []
    y_train = []

    ############## Task 1
    
    ##################
    for i in range(n_train):
        n_samples = np.random.randint(10)
        x = [np.random.randint(10) if i < n_samples  else 0 for i in range(max_train_card)]
        shuffle(x)
        X_train.append(x)
        y_train.append(sum(x))
            
    ##################

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print("Xtrain shape = ",X_train.shape)
    
    return X_train,y_train


def create_test_dataset():
	
    ############## Task 2
    n_test = 200000
    max_test_card = 100
    X_test = []
    y_test = []
    cards = [5*i for i in range(1,21)]
    
    ##################
    for card in cards :
        x = np.random.randint(10,size=(10000,card))
        y = np.sum(x,axis=1)
        
        X_test.append(x)
        y_test.append(y)
    ##################

    return X_test, y_test