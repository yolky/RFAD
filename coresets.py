import numpy as np
import matplotlib.pyplot as plt
import data
import random_features
import utils
import models
import torch_optimizer as optim

import torch
import torch.nn as nn
import training
from functools import partial
import math
import time
import os
import distillation
import argparse
import sys


#Note most of the stuff in this folder is unused in the the paper

def get_var_reduc_coreset(K, n_samples, do_update = True):
    selected_indices = []
    K = np.copy(K)
    last_trace = np.trace(K)
    diagonal_variances = np.copy(np.diag(K))
    for i in range(n_samples):
        
        scores = (np.sum(K**2, 0)/np.diag(K)) - np.diag(K) #reduction in total trace
        
        
        scores[selected_indices] = -np.inf
        selected_index = np.argmax(scores)
        selected_indices.append(selected_index)
        
        if do_update:
            r1 = K[:, selected_index:selected_index + 1]        
            K = K - (r1 @ r1.T)/K[selected_index, selected_index] 
            last_trace = np.trace(K)
        
    return selected_indices

def get_mmd_coreset(K, n_samples):
    selected_indices = []
    self_scores = 0
    for i in range(n_samples):
        diag_scores = np.diag(K)
        
        other_scores = 2 * np.sum(K, 0)/K.shape[0]
        if i > 0:
            self_scores = 2 * np.sum(K[selected_indices], 0) + diag_scores
            self_scores = self_scores/(i+1)
        else:
            self_scores = diag_scores

        scores = other_scores - self_scores
        scores[selected_indices] = -np.inf
        
        selected_indices.append(np.argmax(scores))
    
    return selected_indices

def get_class_kernel(X_train, class_indices, n_models = 16, k = 2, seed = 0):
    fixed_seed = (np.abs(seed) + 1) * np.array(list(range(n_models)))
    # print(fixed_seed)
    model_class = partial(models.ConvNet_wide, X_train.shape[1], net_norm = 'none', im_size=(X_train.shape[2],X_train.shape[3]), k = 2)
    X_train_features, _ = random_features.get_random_features(X_train[class_indices], model_class, n_models, 4096)
    return X_train_features @ X_train_features.T

def make_coreset(X_train, y_train, samples_per_class, n_classes, strategy, seed = 0):
    coreset_size = samples_per_class * n_classes
    torch.manual_seed(seed)
    X_coreset = torch.empty((coreset_size, *X_train.shape[1:])).normal_(0, 1)
    np.random.seed(seed)
    if strategy == 'noise':
        return X_coreset.cuda()
    else:
        all_indices = []
        for c in range(n_classes):
            class_indices = np.where(y_train.numpy() == c)[0]
            if strategy == 'mmd':
                #thin if larger than >10000 per class (this only happens for split mnist)
                if len(class_indices) > 10000:
                    class_indices = np.array(class_indices)[np.random.choice(len(class_indices), 10000, replace = False)]
                
                K = get_class_kernel(X_train, class_indices, seed=seed)
                selected_indices = get_mmd_coreset(K, samples_per_class)
            elif strategy == 'var':
                if len(class_indices) > 10000:
                    class_indices = np.array(class_indices)[np.random.choice(len(class_indices), 10000, replace = False)]
                    
                K = get_class_kernel(X_train, class_indices, seed=seed)
                selected_indices = get_var_reduc_coreset(K, samples_per_class)
            elif strategy == 'random':
                selected_indices = np.random.choice(len(class_indices), samples_per_class, replace = False)
            else:
                print("unrecognized initialization strategy: {}".format(strategy))
                sys.exit()
                
            all_indices.extend(class_indices[selected_indices])
            
        X_coreset.data = X_train[all_indices].cuda()
    
    return X_coreset