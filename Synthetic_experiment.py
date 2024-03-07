import numpy as np
import matplotlib.pyplot as plt
import networkx as ntx


### Initial defs ###
def coord2ind(coord,size):
    i, j = coord
    m, n = size
    return i*n + j

def evaluate_preds_regr(Y_pred, Y, mask = []):
    if mask==[]:
        Y_pred_masked = Y_pred
        Y_masked = Y
    else:
        Y_pred_masked = Y_pred[mask,:]
        Y_masked = Y[mask,:]
    return np.mean((Y_pred_masked-Y_masked)**2)

def addBias(x):
    return np.concatenate( (x,np.ones((x.shape[0],1))) ,axis=1)

def my_colormap(t):
    #Piesewise Linear Colormap from Red to Green
    R = 1 if t<1/2 else 2*(1-t)
    G = 1 if t>1/2 else 2*t
    B = 0
    return (R,G,B)

np.random.seed(123)


### Adjacency matrix ###
N1 = 30
N2 = 30
N = N1*N2
A = np.zeros((N,N))
for n1 in range(N1):
    for n2 in range(N2):
        if n2<=N2-2:
            A[ coord2ind((n1,n2),(N1,N2)) , coord2ind((n1,n2+1),(N1,N2))  ] = 1
        if n1<=N1-2:
            A[ coord2ind((n1,n2),(N1,N2)) , coord2ind((n1+1,n2),(N1,N2))  ] = 1
A += A.T
G = ntx.from_numpy_matrix(A)

### Features ###
noise = 0.02
coords1 = np.linspace(-1,1,N1).reshape((N1,1))
coords2 = np.linspace(-1,1,N2).reshape((N2,1))
Npos = np.concatenate(( np.kron(coords1,np.ones((N2,1))) , np.tile(coords2,(N1,1)) ),axis=1)
Npos = Npos + np.random.normal(size=(N,2))*noise
X = addBias(Npos)
N, C = X.shape

### Targets ###
T = np.zeros((N,1))
T[X[:,0]>0,0] = 1
F = T.shape[1]

### Dataset split ###
train_rate = 0.8
test_rate  = 0.1
val_rate   = 0.1

idx = np.random.permutation(N)
cut1 = round(train_rate*N)
cut2 = cut1 + round(val_rate*N)

train_idx, val_idx, test_idx = idx[:cut1], idx[cut1:cut2], idx[cut2:]

train_mask = np.zeros(N).astype(bool)
val_mask = np.zeros(N).astype(bool)
test_mask = np.zeros(N).astype(bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

X_train, X_val, X_test = X[train_mask,:], X[val_mask,:], X[test_mask,:]
T_train, T_val, T_test = T[train_mask,:], T[val_mask,:], T[test_mask,:]

### Normalize adjacency matrix ###
A_tilde = A + np.eye(N)
D_tilde = np.diag( np.sum(A_tilde,axis=1) )
D_tilde_to_the_minus_half = np.diag( np.diag(D_tilde)**(-0.5) )
A_hat = D_tilde_to_the_minus_half @ A_tilde @ D_tilde_to_the_minus_half

### ELM-GCN ###
Hs = np.arange(1,1+N)
gammas = 10.**np.arange(-8,1+8)
f = lambda x : 1/(1+np.exp(-x))
print('\nELM-GCN')
try:
    E_train = np.load('data/synthetic_ELM_GCN_train.npy')
    E_val   = np.load('data/synthetic_ELM_GCN_val.npy'  )
    print('Loaded')
except:
    print('Training')
    E_train = np.zeros( Hs.shape[0] )
    E_val   = np.zeros( Hs.shape[0] )
    for (idxH,H) in enumerate(Hs):
        if H%50==0: print('H = '+str(H))
        W1 = np.random.rand(C,H)*20-10
        Yh = f(A_hat @ X @ W1)
        AY = A_hat @ Yh
        AY_train = AY[train_mask,:]
        W2 = np.linalg.pinv(AY_train) @ T_train
        Y  = AY @ W2
        
        E_train[idxH] = evaluate_preds_regr(Y, T, train_mask)
        E_val  [idxH] = evaluate_preds_regr(Y, T, val_mask  )
    np.save('data/synthetic_ELM_GCN_train.npy',E_train)
    np.save('data/synthetic_ELM_GCN_val.npy'  ,E_val  )

### RELM-GCN ###
print('\nRELM-GCN')
try:
    E_train2 = np.load('data/synthetic_RELM_GCN_train.npy')
    E_val2   = np.load('data/synthetic_RELM_GCN_val.npy'  )
    print('Loaded')
except:
    print('Training')
    E_train2 = np.zeros( (gammas.shape[0],Hs.shape[0]) )
    E_val2   = np.zeros( (gammas.shape[0],Hs.shape[0]) )
    for (idxH,H) in enumerate(Hs):
        if H%10==0: print('H = '+str(H))
        W1 = np.random.rand(C,H)*20-10
        Yh = f(A_hat @ X @ W1)
        AY = A_hat @ Yh
        AY_train = AY[train_mask,:]
        for (idxgamma,gamma) in enumerate(gammas):
            W2 = np.linalg.inv( 1/gamma*np.eye(H) + AY_train.T @ AY_train ) @ AY_train.T @ T_train
            Y  = AY @ W2
            
            E_train2[idxgamma,idxH] = evaluate_preds_regr(Y, T, train_mask)
            E_val2  [idxgamma,idxH] = evaluate_preds_regr(Y, T, val_mask  )
    np.save('data/synthetic_RELM_GCN_train.npy',E_train2)
    np.save('data/synthetic_RELM_GCN_val.npy',E_val2)

### Plot hyperparameters sweep ###
plt.figure()

### ELM-GCN ###
plt.subplot(121)
plt.plot(Hs,E_train,'r',linewidth=2)
plt.plot(Hs,E_val  ,'b',linewidth=2)
plt.xlabel('Number of Hidden Neurons',fontsize=10)
plt.xticks([0,250,500,750,1000],fontsize=10)
plt.legend(('Training','Validation'),fontsize=10)
plt.ylim([0,.1])
plt.ylabel('MSE',fontsize=12)
plt.yticks([0,.025,.05,.075,.1],fontsize=10)
plt.title('ELM-GCN',fontsize=10)

### RELM-GCN ###
plt.subplot(122)
plt.plot(Hs,np.min(E_train2,axis=0),'r',linewidth=2)
plt.plot(Hs,np.min(E_val2,axis=0)  ,'b',linewidth=2)
plt.xlabel('Number of Hidden Neurons',fontsize=10)
plt.xticks([0,250,500,750,1000],fontsize=10)
plt.legend(('Training','Validation'),fontsize=10)
plt.ylim([0,.1])
plt.ylabel('MSE',fontsize=10)
plt.yticks([0,.025,.05,.075,.1],fontsize=10)
plt.title('RELM-GCN',fontsize=10)

### Hyperparameter selection ###
idxH_star = np.where( E_val==np.min(E_val) )[0][0]
H_star = Hs[idxH_star]

W1 = np.random.rand(C,H_star)*20-10
Yh = f(A_hat @ X @ W1)
AY = A_hat @ Yh
AY_train = AY[train_mask,:]
W2 = np.linalg.pinv(AY_train) @ T_train
Y_ELM_GCN = AY @ W2

Y_ELM_GCN_th = Y_ELM_GCN.copy()
Y_ELM_GCN_th[Y_ELM_GCN_th< 0.5] = 0
Y_ELM_GCN_th[Y_ELM_GCN_th>=0.5] = 1

idx_gamma_star, idx_H_star = np.where( E_val2==np.min(E_val2) )
idx_gamma_star, idx_H_star = idx_gamma_star[0]     , idx_H_star[0]
gamma_star    , H_star     = gammas[idx_gamma_star], Hs[idx_H_star]

W1 = np.random.rand(C,H_star)*20-10
Yh = f(A_hat @ X @ W1)
AY = A_hat @ Yh
AY_train = AY[train_mask,:]
W2 = np.linalg.inv( 1/gamma_star*np.eye(H_star) + AY_train.T @ AY_train ) @ AY_train.T @ T_train
Y_RELM_GCN = AY @ W2

Y_RELM_GCN_th = Y_RELM_GCN.copy()
Y_RELM_GCN_th[Y_RELM_GCN_th< 0.5] = 0
Y_RELM_GCN_th[Y_RELM_GCN_th>=0.5] = 1

### Plots graphs ###
plt.figure()
blackcolor = min( (0,np.min(Y_ELM_GCN),np.min(Y_RELM_GCN)) )
whitecolor = max( (1,np.max(Y_ELM_GCN),np.max(Y_RELM_GCN)) )

T_color             = (T            -blackcolor)/(whitecolor-blackcolor)
Y_ELM_GCN_color     = (Y_ELM_GCN    -blackcolor)/(whitecolor-blackcolor)
Y_ELM_GCN_th_color  = (Y_ELM_GCN_th -blackcolor)/(whitecolor-blackcolor)
Y_RELM_GCN_color    = (Y_RELM_GCN   -blackcolor)/(whitecolor-blackcolor)
Y_RELM_GCN_th_color = (Y_RELM_GCN_th-blackcolor)/(whitecolor-blackcolor)

plt.subplot(231)
plt.xlabel('Ground Truth',fontsize=10)
plt.xticks([])
plt.yticks([])
ntx.draw_networkx_edges(G, pos=Npos, edge_color='black', alpha=1)
for n in range(N):
    plt.plot(X[n,0],X[n,1],'ok',markerfacecolor=my_colormap(T_color[n,0]),markersize=8)


plt.subplot(232)
plt.xlabel('ELM-GCN',fontsize=10)
plt.xticks([])
plt.yticks([])
ntx.draw_networkx_edges(G, pos=Npos, edge_color='black', alpha=1)
for n in range(N):
    plt.plot(X[n,0],X[n,1],'ok',markerfacecolor=my_colormap(Y_ELM_GCN_color[n,0]),markersize=8)

plt.subplot(233)
plt.xlabel('ELM-GCN (Thresholded)',fontsize=10)
plt.xticks([])
plt.yticks([])
ntx.draw_networkx_edges(G, pos=Npos, edge_color='gray', alpha=1)
for n in range(N):
    plt.plot(X[n,0],X[n,1],'ok',markerfacecolor=my_colormap(Y_ELM_GCN_th_color[n,0]),markersize=8)

plt.subplot(235)
plt.xlabel('RELM-GCN',fontsize=10)
plt.xticks([])
plt.yticks([])
ntx.draw_networkx_edges(G, pos=Npos, edge_color='gray', alpha=1)
for n in range(N):
    plt.plot(X[n,0],X[n,1],'ok',markerfacecolor=my_colormap(Y_RELM_GCN_color[n,0]),markersize=8)

plt.subplot(236)
plt.xlabel('RELM-GCN (Thresholded)',fontsize=10)
plt.xticks([])
plt.yticks([])
ntx.draw_networkx_edges(G, pos=Npos, edge_color='gray', alpha=1)
for n in range(N):
    plt.plot(X[n,0],X[n,1],'ok',markerfacecolor=my_colormap(Y_RELM_GCN_th_color[n,0]),markersize=8)



    





















