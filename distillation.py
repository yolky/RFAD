import numpy as np
import matplotlib.pyplot as plt
import data
import random_features
import utils
from utils import double_print
import models
import torch_optimizer as torch_optim
import random
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import training
from functools import partial
import math
import time
import os

def corrupt_data(X, corruption_mask, X_init, whitening_mat):
    # return X
    X_corrupted = ((1 - corruption_mask) * X) + (corruption_mask * X_init)

    if not whitening_mat is None:
        print('slls')
        X_corrupted = data.transform_data(X_corrupted, whitening_mat)
    return X_corrupted

def distill_dataset(X_train, y_train, model_class, lr, n_models, n_batches, iters = 10000, platt = False, ga_steps = 1, schedule = None, save_location = None, samples_per_class = 1, n_classes = 10, learn_labels = False, batch_size = 1280, X_valid = None, y_valid = None, n_channels = 3, im_size = 32, X_init = None, jit = 1e-6, seed = 0, corruption = 0, whitening_mat = None, from_loader = False):
    coreset_size = samples_per_class * n_classes

    X_coreset = torch.nn.Parameter(torch.empty((coreset_size, n_channels, im_size,im_size), device = 'cuda:0').normal_(0, 1))
    
    transform_mat = torch.nn.Parameter(torch.empty((n_channels * im_size * im_size, n_channels * im_size * im_size), device = 'cuda:0').normal_(0, 1))
    transform_mat.data = torch.eye(n_channels * im_size * im_size, device = 'cuda:0')
    
    k = torch.nn.Parameter(torch.tensor((0.), device = 'cuda:0').double()) #parameter used for platt scaling
    
    y_coreset = torch.nn.Parameter(torch.empty((coreset_size, n_classes), device = 'cuda:0').normal_(0, 1))
    y_coreset.data = (torch.Tensor(utils.one_hot(np.concatenate([[j for i in range(samples_per_class)]  for j in range(n_classes)]), n_classes)).float().cuda() - 1/n_classes)
    
    
    if not X_init is None:
        X_coreset.data = X_init.cuda()
    else:
        X_init = X_coreset.data.clone()
    X_init = X_init.cuda().clone()
    
    if not whitening_mat is None:
        whitening_mat = whitening_mat.cuda()
        
    if corruption > 0:
        torch.manual_seed(seed)
        corruption_mask = (torch.rand(size=X_coreset.shape) < corruption).int().float().cuda() # 0 = don't corrupt, 1 = corrupt
            
                
    losses = []

    if platt:
        if not learn_labels:
            optim = torch_optim.AdaBelief([{"params": [X_coreset]},
              {"params": [transform_mat], "lr": 5e-5}, {"params": [k], "lr": 1e-2}], lr = lr, eps = 1e-16) #a larger learning rate for k usually helps
        else:
            optim = torch_optim.AdaBelief([{"params": [X_coreset, y_coreset]},
              {"params": [transform_mat], "lr": 5e-5}, {"params": [k], "lr": 1e-2}], lr = lr, eps = 1e-16)
    else:
        if not learn_labels:
            optim = torch_optim.AdaBelief([{"params": [X_coreset]},
              {"params": [transform_mat], "lr": 5e-5}], lr = lr, eps = 1e-16)
        else:
            optim = torch_optim.AdaBelief([{"params": [X_coreset, y_coreset]},
              {"params": [transform_mat], "lr": 5e-5}], lr = lr, eps = 1e-16)

    model_rot = 10
    schedule_i = 0
    
    valid_fixed_seed = (np.abs(seed) + 1) * np.array(list(range(16)))
    
    if X_valid is not None:
        X_valid_features, _ = random_features.get_random_features(X_valid, model_class, 16, 4096, fixed_seed = valid_fixed_seed)
        y_valid_one_hot = utils.one_hot(y_valid, n_classes) - 1/n_classes
    X_coreset_best = None
    y_coreset_best = None
    k_best = None
    best_iter = -1
    best_valid_loss = np.inf
    acc = 0
    
    start_time = time.time()
    output_file = None
    
    if save_location is not None:
        if not os.path.isdir(save_location):
            os.makedirs(save_location)
        output_file = open('{}/training_log.txt'.format(save_location) ,'a')

    file_print = partial(double_print, output_file = output_file)
    
    
    if from_loader:
        X_iterator = iter(X_train)
    
    for i in range(iters):
        if i%(ga_steps * 40) == 0:
            file_print(acc)
            transformed_coreset = data.transform_data(X_coreset.data, transform_mat.data)
            
            if corruption > 0:
                transformed_coreset = corrupt_data(transformed_coreset, corruption_mask, X_init, whitening_mat)
            
            if save_location is not None:
                np.savez('{}/{}.npz'.format(save_location,i), images = transformed_coreset.data.cpu().numpy(), labels = y_coreset.data.cpu().numpy(), k=k.data.cpu(), jit = jit)
            
            #get validation acc
            
            X_coreset_features, _ = random_features.get_random_features(transformed_coreset.cpu(), model_class, 16, 4096, fixed_seed = valid_fixed_seed)
            K_xx = 2 * (X_coreset_features @ X_coreset_features.T) + 0.01
            K_xx = K_xx + (jit * np.eye(1 * coreset_size) * np.trace(K_xx)/coreset_size)
            solved = np.linalg.solve(K_xx.astype(np.double), y_coreset.data.cpu().numpy().astype(np.double))
            preds_valid = (2 * (X_valid_features @ X_coreset_features.T) + 0.01).astype(np.double) @ solved

            if not platt:
                valid_loss = 0.5 * np.mean((y_valid_one_hot - preds_valid)**2)
                
            else:
                valid_loss = nn.CrossEntropyLoss()(torch.exp(k) * torch.tensor(preds_valid).cuda(), y_valid.cuda()).detach().cpu().item()
            valid_acc = np.mean(preds_valid.argmax(axis = 1) == y_valid_one_hot.argmax(axis = 1))
            file_print('iter: {}, valid loss: {}, valid acc: {}, elapsed time: {:.1f}s'.format(i, valid_loss, valid_acc, time.time() - start_time))
            
            if valid_loss < best_valid_loss:
                X_coreset_best = X_coreset.data.detach().clone()
                transform_mat_best = transform_mat.data.detach().clone()
                y_coreset_best = y_coreset.data.detach().clone()
                k_best = k.data.detach().clone()
                best_iter = i
                best_valid_loss = valid_loss
            
            patience = 1000
            if (i > best_iter + (ga_steps * patience) and i > schedule[-1][0] + (ga_steps * patience)) or iters == 1:
                file_print('early stopping at iter {}, reverting back to model from iter {}'.format(i, best_iter))
                transformed_best_coreset = data.transform_data(X_coreset_best.data, transform_mat_best.data)
                
                if corruption > 0:
                    transformed_best_coreset = corrupt_data(transformed_best_coreset, corruption_mask, X_init, whitening_mat)
                    
                np.savez('{}/best.npz'.format(save_location), images = transformed_best_coreset.data.cpu().numpy(), labels = y_coreset_best.data.cpu().numpy(), valid_loss = best_valid_loss, k = k_best.data.cpu().numpy(), jit = jit, best_iter = best_iter)
                return transformed_best_coreset, y_coreset_best.data
            
        
        if schedule is not None and schedule_i < len(schedule):
            if i >= schedule[schedule_i][0]:
                file_print("UPDATING MODEL COUNT: {}".format(schedule[schedule_i]))
                n_models = schedule[schedule_i][1]
                model_rot = schedule[schedule_i][2]
                schedule_i += 1
        
        
        if i % ga_steps == 0:
            optim.zero_grad()
            
        if i%model_rot == 0:
            if i != 0:
                del models_list
                
            models_list = []
            rand_seed = random.randint(0, 50000)
            torch.manual_seed(rand_seed)
            for m in range(n_models):
                models_list.append(model_class(n_random_features = 4096, chopped_head = True))

                models_list[-1].to('cuda:0')
                models_list[-1].eval()

        X_coreset_features = []
        transformed_data = data.transform_data(X_coreset, transform_mat)
        if corruption > 0:
            transformed_data = corrupt_data(transformed_data, corruption_mask, X_init, whitening_mat)
                
        for m in range(n_models):
            X_coreset_features.append(models_list[m](transformed_data))
        X_coreset_features = torch.cat(X_coreset_features, 1)/np.sqrt(n_models * X_coreset_features[0].shape[1])

        K_xx = (2 * X_coreset_features @ X_coreset_features.T) + 0.01
        K_xx = K_xx + (jit * torch.eye(1 * coreset_size, device = 'cuda:0') * torch.trace(K_xx)/coreset_size)
        
        X_train_features = []
        y_values = []
        with torch.no_grad():
            for b in range(n_batches):
                if not from_loader:
                    indices = np.random.choice(X_train.shape[0], 1280, replace = False)
                    X_batch = X_train[indices].float().cuda()
                    y_batch = y_train[indices]
                else:
                    try:
                        batch = next(X_iterator)
                    except StopIteration:
                        X_iterator = iter(X_train)
                        batch = next(X_iterator)
                    X_batch = batch[0].cuda()
                    y_batch = batch[1]
                
                X_train_features_inner = []

                for m in range(n_models):
                    X_train_features_inner.append(models_list[m](X_batch).detach())

                y_values.append(torch.nn.functional.one_hot(y_batch, n_classes).cuda() - 1/n_classes)

                X_train_features_inner = torch.cat(X_train_features_inner, 1)/np.sqrt(n_models * X_train_features_inner[0].shape[1])
                
                X_train_features.append(X_train_features_inner)


            X_train_features = torch.cat(X_train_features, 0).detach()
            y_values = torch.cat(y_values, 0)
        
        
        solved = torch.linalg.solve(K_xx.double(), y_coreset.double())
        K_zx = 2 * (X_train_features @ X_coreset_features.T) + 0.01
        preds = K_zx.double() @ solved
        
                
        acc = np.mean(preds.detach().cpu().numpy().argmax(axis = 1) == y_values.cpu().numpy().argmax(axis = 1))
        
        if platt:
            loss = nn.CrossEntropyLoss()(torch.exp(k) * preds, torch.argmax(y_values, 1))
        else:
            loss = .5 * torch.mean((y_values - preds)**2)
        
        if i % ga_steps == (ga_steps - 1):
            loss.backward()

        losses.append((loss).detach().cpu().numpy().item())
        
        if i % ga_steps == (ga_steps - 1):
            
            optim.step()    
            file_print('=', end = '')