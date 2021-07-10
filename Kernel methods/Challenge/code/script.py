import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#import kernels
from string_kernels import mismatch_kernel
from real_kernels import rbf_kernel


SEED = 42
np.random.seed(SEED)

# loading sequences
def loading_sequence(path):
    seq = pd.read_csv(path)
    seq = seq.to_numpy()[:,1]
    return seq

    
    
Xtr0 = loading_sequence("./data/Xtr0.csv")
Xtr1 = loading_sequence("./data/Xtr1.csv")
Xtr2 = loading_sequence("./data/Xtr2.csv")

# loading labels
def loading_labels(path):
    label = pd.read_csv(path)
    label = label.to_numpy()[:,1]
    label[label==0] = -1
    return label
    
ytr0 = loading_labels("./data/Ytr0.csv")
ytr1 = loading_labels("./data/Ytr1.csv")
ytr2 = loading_labels("./data/Ytr2.csv")



# loading float values for sequences
def loading_sequence_float(path):
    seq_mat = pd.read_csv(path, header=None)
    seq_mat = seq_mat.to_numpy()
    seq_mat = np.array([[float(v) for v in r[0].split(' ')] for r in seq_mat])
    return seq_mat
    
Xtr0_mat = loading_sequence_float("./data/Xtr0_mat100.csv")
Xtr1_mat = loading_sequence_float("./data/Xtr1_mat100.csv")
Xtr2_mat = loading_sequence_float("./data/Xtr2_mat100.csv")


# loading test set
Xte0 = loading_sequence("./data/Xte0.csv")
Xte1 = loading_sequence("./data/Xte1.csv")
Xte2 = loading_sequence("./data/Xte2.csv")

Xte0_mat = loading_sequence_float("./data/Xte0_mat100.csv")
Xte1_mat = loading_sequence_float("./data/Xte1_mat100.csv")
Xte2_mat = loading_sequence_float("./data/Xte2_mat100.csv")




# compute alpha for kernel ridge regression
def compute_alpha(K, y, λ=.005):
    """
    Returns the predited label
    """
    n = K.shape[0]
    alpha = np.linalg.solve(K + λ * n * np.eye(n), y)
    return alpha


def estimate(train_size, Kernel, alpha):
    """
    Returns the predited label
    """
    L = []
    n = Kernel.shape[0]
    for i in range(train_size, n):
        L.append(np.sum(alpha * Kernel[i, :train_size]))
    return np.array(L)


def accuracy(Y_predited, Y_true):
    return (Y_predited==Y_true).mean()
    
    

    
# predictions
def return_prediction(X_train, X_train_mat, y_train, X_test, X_test_mat, ksize_spec, ksize_spec_2,ksize_mis, sigma=1, λ=.05):
    X_tot = np.concatenate((X_train, X_test))
    X_tot_mat = np.concatenate((X_train_mat, X_test_mat))
    train_size = X_train.size
    
    # rbf
    kernel_rbf = rbf_kernel(X_tot_mat, sigma=sigma)
    
    # spectrum
    kernel_spec = mismatch_kernel(X_tot,ksize_spec,m=0)
    kernel_spec_2 = mismatch_kernel(X_tot,ksize_spec_2,m=0)
    
    # mismatch
    kernel_mis = mismatch_kernel(X_tot, ksize_mis, m=1)
    
    print(kernel_spec.shape, kernel_mis.shape, kernel_rbf.shape, kernel_spec_2.shape)
    # sum kernel
    kernel = kernel_spec + kernel_mis #+ kernel_rbf + kernel_spec_2
    
    alpha = compute_alpha(kernel[:train_size, :train_size], y_train, λ=λ)

    y_predict = estimate(train_size, kernel, alpha)
    y_predict = np.array(y_predict)
    
    y_predict[y_predict > 0] = 1
    y_predict[y_predict < 0] = 0
    
    return y_predict
    
    
# create and save dataframe
def save_score(y_predict, name_prediction):
    size_predict = y_predict.size
    h = np.arange(0, size_predict, 1)
    y = pd.DataFrame({'Id': h, 'Bound': y_predict})
    y.to_csv(name_prediction, index=False)
    


if __name__ == '__main__':
    
    
    X_train0, X_val0, y_train0, y_val0 = train_test_split(Xtr0, ytr0, test_size=0.1, random_state=42)
    X_train1, X_val1, y_train1, y_val1 = train_test_split(Xtr1, ytr1, test_size=0.1, random_state=42)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(Xtr2, ytr2, test_size=0.1, random_state=42)

    X_train0_mat, X_val_mat0, _, _ = train_test_split(Xtr0_mat, ytr0, test_size=0.1, random_state=SEED)
    X_train1_mat, X_val_mat1, _, _ = train_test_split(Xtr1_mat, ytr1, test_size=0.1, random_state=SEED)
    X_train2_mat, X_val_mat2, _, _ = train_test_split(Xtr2_mat, ytr2, test_size=0.1, random_state=SEED)

    y_predict_0 = return_prediction(X_train0, X_train0_mat, y_train0, Xte0, Xte0_mat, 3, 2, 7, sigma=10, λ=0.0009545006745295772)
    y_predict_1 = return_prediction(X_train1, X_train1_mat, y_train1, Xte1, Xte1_mat, 9, 4, 6, sigma=10, λ=7.937037902169388e-06)
    y_predict_2 = return_prediction(X_train2, X_train2_mat, y_train2, Xte2, Xte2_mat, 9, 3, 8, λ=0.0004722502785762801)
    y = np.concatenate((y_predict_0, y_predict_1, y_predict_2))
    y = np.array(y, dtype=np.int)
    save_score(y, "y_predicted.csv")
