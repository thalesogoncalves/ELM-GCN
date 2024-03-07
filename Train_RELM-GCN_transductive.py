import numpy as np
from utils import load_data
import time
import scipy.sparse as sp

### Define dataset, number of hidden units and regularization parameter ###
dataset, H, gamma = 'cora'    , 1000, 10.**(-5)
#dataset, H, gamma = 'citeseer', 1000, 10.**(-6)
#dataset, H, gamma = 'pubmed'  , 1250, 10.**(-3)
#dataset, H, gamma = 'reddit'  , 1500, 10.**( 1)
num_executions = 10

### Initial defs ###
def evaluate_preds_class(Y_pred, Y, mask = []):
    if mask==[]:
        Y_pred_masked = Y_pred
        Y_masked = Y
    else:
        Y_pred_masked = Y_pred[mask,:]
        Y_masked = Y[mask,:]
    N = Y_masked.shape[0]
    labels_pred = np.argmax(Y_pred_masked, axis=1).reshape(N)
    labels = np.argmax(Y_masked, axis=1).reshape(N)
    return np.mean( labels_pred==labels )

def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def loadRedditFromNPZ():
    adj = sp.load_npz("data/reddit_adj.npz")
    data = np.load("data/reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def transferLabel2Onehot(labels, N):
    y = np.zeros((len(labels),N))
    for i in range(len(labels)):
        pos = labels[i]
        y[i,pos] =1
    return y

f = lambda x : np.multiply(x,x>0) # ReLU
np.random.seed(123)

### Load dataset ###
print('Loading Dataset...', end='')
if dataset=='reddit':
    A, X, T_train, T_val, T_test, train_idx, val_idx, test_idx = loadRedditFromNPZ()
else:
    A, X, T_train, T_val, T_test, train_idx, val_idx, test_idx = load_data(dataset)
print(' Ok!')

### Pre-process data ###
print('Pre-Processing...', end='')
if dataset=='reddit':
    A = A+A.T
    X = nontuple_preprocess_features(X)
    
    T_train = transferLabel2Onehot(T_train, 41)
    T_val   = transferLabel2Onehot(T_val  , 41)
    T_test  = transferLabel2Onehot(T_test , 41)    
else:
    T_train = T_train[train_idx,:]
    T_val   = T_val  [val_idx  ,:]
    T_test  = T_test [test_idx ,:]

A_hat = nontuple_preprocess_adj(A)
N, C = X.shape
print(' Ok!')

### RELM-GCN ###
print('\nTraining Networks')
acc_tests   = np.zeros( num_executions )
train_times = np.zeros( num_executions )
for execution in range(num_executions):
    print('   Training   {}/{}...'.format(1+execution,num_executions), end='')
    t0 = time.time()
    W1 = np.random.rand(C,H)*20-10
    Yh = f(A_hat @ X @ W1)
    AY = A_hat[train_idx,:] @ Yh
    W2 = np.linalg.inv( 1/gamma*np.eye(H) + AY.T @ AY ) @ AY.T @ T_train
    t  = time.time()
    train_times[execution] = t-t0
    print(' Ok! (Training Time = {:.2f} s)'.format(train_times[execution]))
    
    print('   Evaluating {}/{}...'.format(1+execution,num_executions), end='')
    Y_test  = A_hat[test_idx,:] @ Yh @ W2
    acc_tests[execution] = evaluate_preds_class(Y_test, T_test)
    print(' Ok! (Test Accuracy = {:.4f})\n'.format(acc_tests[execution]))
    del W1, Yh, AY, W2, Y_test

print('Average Test Accuracy: {:.4f}'.format(np.mean(acc_tests)))
print('Average Training Time: {:.2f} s'.format(np.mean(train_times)))


















