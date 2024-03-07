import numpy as np
from scipy.stats import wilcoxon

### Parameters ###
modes      = np.array(['Transductive', 'Inductive'])
datasets   = np.array(['Cora', 'Citeseer', 'Pubmed', 'Reddit'])
algorithms = np.array(['GCN', 'fastGCN', 'ELM-GCN', 'RELM-GCN'])
trials     = np.arange(1,1+10)

num_modes      = modes.shape[0]
num_datasets   = datasets.shape[0]
num_algorithms = algorithms.shape[0]
num_trials     = trials.shape[0]

folder_name        = 'data/train_hist'
file_name_template = 'algorithm_mode_dataset_trial'

times      = np.zeros( (num_modes, num_datasets, num_algorithms, num_trials ) )
accuracies = np.zeros( (num_modes, num_datasets, num_algorithms, num_trials ) )
for (mode_idx,mode) in enumerate(modes):
    for (dataset_idx,dataset) in enumerate(datasets):
        for (algorithm_idx,algorithm) in enumerate(algorithms):
            if algorithm=='GCN' and dataset=='Reddit': continue
            for trial in trials:
                trial_idx = trial-1
                file_name = file_name_template.replace('mode', mode.lower()
                                             ).replace('dataset', dataset.lower()
                                             ).replace('algorithm', algorithm
                                             ).replace('trial', str(trial))
                train_hist = np.load('{}/{}.npy'.format(folder_name,file_name))
                times     [mode_idx,dataset_idx,algorithm_idx,trial_idx] = train_hist[-1,0]
                accuracies[mode_idx,dataset_idx,algorithm_idx,trial_idx] = train_hist[-1,1]

### Wilcoxon hypothesis test ###
alpha = 0.001 # Significance 99.9%
var_names = ['Test Accuracy', 'Training Time']
for (var_idx,var) in enumerate([accuracies, times]):
    print('\n\n\n{} Analysis'.format(var_names[var_idx]))
    for (mode_idx,mode) in enumerate(modes):
        print('\n{}'.format(mode))
        for (algorithm_idx1,algorithm1) in enumerate(algorithms):
            for (algorithm_idx2,algorithm2) in enumerate(algorithms):
                if algorithm_idx1 < algorithm_idx2:
                    if (algorithm1=='GCN') or (algorithm2=='GCN'):
                        var1 = var[mode_idx,:-1,algorithm_idx1,:].flatten()
                        var2 = var[mode_idx,:-1,algorithm_idx2,:].flatten()
                    else:
                        var1 = var[mode_idx,:  ,algorithm_idx1,:].flatten()
                        var2 = var[mode_idx,:  ,algorithm_idx2,:].flatten()
                    stats, p = wilcoxon(var1,var2)
                    print('{} x {}: '.format(algorithm1,algorithm2), end='')
                    if p > alpha:
                        print('The test can\'t reject the hypothesis that {} and {} produces same distriburion (p = {:.4f})'.format(algorithm1,algorithm2,p))
                    else:
                        print('The test rejects the hypothesis that {} and {} produces same distriburion (p = {:.4f}). Moreover, '.format(algorithm1,algorithm2,p), end='')
                        if var_idx==0:
                            stats, p = wilcoxon(var1,var2,alternative='greater')
                            if p > alpha:
                                print('the test can\'t reject the hypothesis that {} is less accurate than {} (p = {:.4f})'.format(algorithm1,algorithm2,p))
                            else:
                                print('the test rejects the hypothesis that {} is less accurate than {} (p = {:.4f})'.format(algorithm1,algorithm2,p))
                        else:
                            stats, p = wilcoxon(var1,var2,alternative='less')
                            if p > alpha:
                                print('the test can\'t reject the hypothesis that {} is slower than {} (p = {:.4f})'.format(algorithm1,algorithm2,p))
                            else:
                                print('the test rejects the hypothesis that {} is slower than {} (p = {:.4f})'.format(algorithm1,algorithm2,p))
                
                
                
                
                
                
                
                
                
                
                
                
                