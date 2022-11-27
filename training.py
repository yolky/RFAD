import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import models
import copy
import math
import data
from diffaugment import ParamDiffAug, DiffAugment

def train_network(X_train, y_train, X_valid, y_valid, net_width, max_iters, lr, weight_decay, loss_mode, centering, patience = 500, log = False, batch_size = 128, label_scale_factor = 8, seed = None, data_aug = False, from_loader = False, im_size = None, n_classes = None, n_channels = None, net_norm = 'none'):
    log = True
    
    if not from_loader:
        n_classes = y_train.shape[1]
        im_size = X_train.shape[-1]
        n_channels = X_train.shape[1]
        
    
    print(n_classes)
    if seed is not None:
        torch.manual_seed(seed)
    model = models.ConvNet(n_channels, n_classes, net_width = net_width, net_norm = net_norm, im_size = (im_size, im_size)).cuda()
    model_init = copy.deepcopy(model)
    
    optim = torch.optim.SGD(list(model.parameters()), lr = lr, momentum = 0.9, weight_decay = 0)

    best_valid_acc = -1
    best_model = None
    best_iter = -1
    
    if from_loader:
        X_iterator = iter(X_train)
    
    for i in range(max_iters):
        optim.zero_grad()
        fac = 1
        if not from_loader:
            if batch_size < X_train.shape[0]:
                batch_indices = np.random.choice(X_train.shape[0], batch_size)
                X_batch = X_train[batch_indices]

                if data_aug:
                    donger = ParamDiffAug()
                    X_batch = DiffAugment(X_batch, strategy = 'color_crop_cutout_scale_rotate', param = donger)
                y_batch = y_train[batch_indices]
            else:
                X_batch = X_train
                y_batch = label_scale_factor * fac * y_train
        else:
            try:
                batch = next(X_iterator)
            except StopIteration:
                X_iterator = iter(X_train)
                batch = next(X_iterator)
            X_batch = batch[0].cuda()
            y_batch = torch.nn.functional.one_hot(batch[1].cuda(), n_classes)
        output = fac * model(X_batch)
        if centering:
            with torch.no_grad():
                output_0 = fac * model_init(X_batch)
        else:
            output_0 = 0
        if loss_mode == 'xent':
            loss = torch.nn.functional.cross_entropy(output - output_0, torch.argmax(y_batch, 1))
        elif loss_mode == 'mse':
            loss = torch.mean((output - output_0 - y_batch)**2)/label_scale_factor**2
        for pa, pb in zip(model.parameters(), model_init.parameters()):
            if centering:
                loss += weight_decay * torch.sum((pa-pb)**2) #+ weight_decay * torch.sum((pc-pb)**2)
            else:
                loss += weight_decay * torch.sum(pa**2)
                            
        if i%100 == 0:
            valid_acc = get_acc(model, model_init, X_valid, y_valid, centering = centering)
            if i == 0:
                valid_acc0 = valid_acc
            if log:
                print('iter: {}, valid_acc: {}'.format(i, valid_acc))
            
            if valid_acc > best_valid_acc:
                best_iter = i
                best_model = copy.deepcopy(model)
                best_valid_acc = valid_acc
            
            model.train()
            
        if i >= best_iter + patience:
            print('early stopping at iter {}'.format(i))
            print('reverting to iter {}'.format(best_iter))
            
            break
            
        loss.backward()
        optim.step()
        
    return best_model, model_init, best_valid_acc
    
def get_acc(model, model_init, X, y, batch_size = 512, centering = True, return_predictions = False, from_loader = False):
    model.eval()
    if model_init is not None:
        model_init.eval()
    n_correct = 0
    predictions = []
    if not from_loader:
        total_count = X.shape[0]
        for i in range(math.ceil(X.shape[0]/batch_size)):
            X_batch = X[batch_size * i: batch_size * (i+1)].cuda()
            y_batch = y[batch_size * i: batch_size * (i+1)].cuda()
            output = model(X_batch)
            if centering:
                output_0 = model_init(X_batch)
            else:
                output_0 = 0
            predictions.append((output - output_0).detach().cpu().numpy())
            n_correct += torch.sum((output - output_0).argmax(1) == y_batch).item()
    else:
        total_count = 0
        for batch in X:
            X_batch = batch[0].cuda()
            y_batch = batch[1].cuda()
            output = model(X_batch)
            if centering:
                output_0 = model_init(X_batch)
            else:
                output_0 = 0
            predictions.append((output - output_0).detach().cpu().numpy())
            n_correct += torch.sum((output - output_0).argmax(1) == y_batch).item()
            total_count += y_batch.shape[0]
    if return_predictions:
        return n_correct/total_count, np.concatenate(predictions, 0)
    return n_correct/total_count