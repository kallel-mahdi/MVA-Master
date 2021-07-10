"""
Learning on Sets - ALTEGRAD - Jan 2021
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        test_batch = X_test[i][j:min(j+batch_size,n_samples_per_card)]
        test_batch = torch.LongTensor(test_batch).to(device)
        y_pred_d = deepsets(test_batch)
        y_pred_deepsets.append(y_pred_d)
        
        y_pred_l = lstm(test_batch)
        y_pred_lstm.append(y_pred_l)
        ##################
        
    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    
    acc_deepsets = accuracy_score(np.round(y_pred_deepsets),y_test[i])
    mae_deepsets = mean_absolute_error(y_pred_deepsets,y_test[i])
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    acc_lstm = accuracy_score(np.round(y_pred_lstm),y_test[i])
    mae_lstm = mean_absolute_error(y_pred_lstm,y_test[i])
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
    
##################

cards = [5*i for i in range(1,21)]
fig = plt.figure()
plt.plot(cards,results['lstm']['acc'],label='LSTM')
plt.plot(cards,results['deepsets']['acc'],label='DEEPSETS')
plt.xlabel("Set cardinality")
plt.ylabel("Accuracy")
plt.legend()
#plt.savefig("/content/test.png")
plt.show();

##################