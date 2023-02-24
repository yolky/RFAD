import numpy as np
import matplotlib.pyplot as plt
import data
import random_features
import utils
from utils import double_print
import models
import torch_optimizer as optim

import torch
import torch.nn as nn
from training import train_network, get_acc
from functools import partial
import math
import time
import os
import distillation
import argparse
import sys
import datetime
import copy
import kernels

import neural_tangents as nt
from neural_tangents import stax
import functools
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_path', type = str)
    parser.add_argument('--epoch', type = str, default = 'best')
    parser.add_argument('--run_krr', action='store_true')
    parser.add_argument('--run_finite', action='store_true')
    parser.add_argument('--valid_seed', type = int, default = 0)
    
    parser.add_argument('--net_width', type = int, default = 1024)
    parser.add_argument('--lr', type = float, default = 1e-2)
    parser.add_argument('--weight_decay', type = float, default = 0)
    parser.add_argument('--loss_mode', type = str, default = 'mse')
    parser.add_argument('--label_scale', type = float, default = 1)
    parser.add_argument('--identifier', type = str, default = '')
    parser.add_argument('--centering', action='store_true')
    parser.add_argument('--no_kernel_save', action='store_true')
    parser.add_argument('--use_best_hypers', action='store_true')

    args = parser.parse_args()

    
    if args.dataset == 'mnist':
        im_size = 28
        n_channels = 1
        n_classes = 10
        X_train, y_train, X_test, y_test  = data.get_mnist(output_channels = 1, image_size = im_size)
    elif args.dataset == 'fashion':
        im_size = 28
        n_channels = 1
        n_classes = 10
        X_train, y_train, X_test, y_test  = data.get_fashion_mnist(output_channels = 1, image_size = im_size)
    elif args.dataset == 'cifar10':
        im_size = 32
        n_channels = 3
        n_classes = 10
        X_train, y_train, X_test, y_test  = data.get_cifar10(output_channels = n_channels, image_size = im_size)
        
        
        whitening_mat = data.get_zca_matrix(X_train, reg_coef = 0.1)
        X_train = data.transform_data(X_train, whitening_mat)
        X_test = data.transform_data(X_test, whitening_mat)
    elif args.dataset == 'cifar100':
        im_size = 32
        n_channels = 3
        n_classes = 100
        X_train, y_train, X_test, y_test  = data.get_cifar100(output_channels = n_channels, image_size = im_size)
        X_start = X_train[0:100].clone()

        whitening_mat = data.get_zca_matrix(X_train, reg_coef = 0.1)
        X_train = data.transform_data(X_train, whitening_mat)
        X_test = data.transform_data(X_test, whitening_mat)
    elif args.dataset == 'svhn':
        im_size = 32
        n_channels = 3
        n_classes = 10
        X_train, y_train, X_test, y_test  = data.get_svhn(output_channels = n_channels, image_size = im_size)
        
        X_train = data.layernorm_data(X_train)
        X_test = data.layernorm_data(X_test)
        
        whitening_mat = data.get_zca_matrix(X_train, reg_coef = 0.1)
        X_train = data.transform_data(X_train, whitening_mat)
        X_test = data.transform_data(X_test, whitening_mat)
    elif args.dataset == 'split_mnist':
        im_size = 28
        n_channels = 1
        n_classes = 2
        X_train, y_train, X_test, y_test  = data.get_mnist(output_channels = 1, image_size = im_size)
        y_train = y_train//5
        y_test = y_test//5
    else:
        print("unrecognized dataset: {}".format(args.dataset))
        sys.exit()
    
    print("Evaluating at path {} on dataset {}".format(args.save_path, args.dataset))
    
    support_set = np.load('{}/{}.npz'.format(args.save_path, args.epoch))
    X_sup = support_set['images']
    y_sup = support_set['labels']
    # print(support_set['k'])
    jit = support_set.get('jit', 5e-3)
    
    np.random.seed(args.valid_seed)
    valid_indices = []
    for c in range(n_classes):
        class_indices = np.where(y_train == c)[0]
        valid_indices.append(class_indices[np.random.choice(len(class_indices), 500 if n_classes == 10 else 100)])

    valid_indices = np.concatenate(valid_indices)
    X_valid = X_train[valid_indices]
    y_valid = y_train[valid_indices]
    
    output_file = open('{}/eval_results_{}.txt'.format(args.save_path, datetime.datetime.now()) ,'a')
    file_print = partial(double_print, output_file = output_file)
    
    if args.run_finite:
        file_print("Running finite results")

        if(args.use_best_hypers):
            args.centering = True
            args.lr, args.label_scale, args.weight_decay = utils.get_best_finite_hypers(args.dataset, X_sup.shape[0]//n_classes, use_label_scale = True)
            print(f"Loading best set of hyperparameters for {args.dataset}, {X_sup.shape[0]//n_classes}:")
            print(f"Learning rate: {args.lr}, label scale: {args.label_scale}, weight decay: {args.weight_decay}")
            
        output_file.write("\n")
        output_file.flush()
        model, model_init, valid_acc = train_network(torch.tensor(X_sup).cuda(), torch.tensor(y_sup).cuda(), X_valid, y_valid, args.net_width, 20000, args.lr, args.weight_decay, args.loss_mode, args.centering, patience = 500, batch_size = 513, label_scale_factor = args.label_scale, seed = args.valid_seed, net_norm = 'none')
        test_acc, test_predictions = get_acc(model, model_init, X_test, y_test, return_predictions = True, centering = args.centering)
        file_print("Centering: {}, loss_mode: {}, lr: {}, weight_decay: {}, label_scale: {}, valid_acc: {}, test_acc: {}\n".format(args.centering, args.loss_mode, args.lr, args.weight_decay, args.label_scale, valid_acc, test_acc))
        
        if len(args.identifier) > 0:
            np.savez('{}/eval_finite_centering_{}_loss_{}_lr_{}_wd_{}_ls_{}_{}.npz'.format(args.save_path, args.centering, args.loss_mode, args.lr, args.weight_decay, args.label_scale, args.identifier), valid_acc = valid_acc, test_acc = test_acc, test_predictions = test_predictions)        
        else:
            np.savez('{}/eval_finite_centering_{}_loss_{}_lr_{}_wd_{}_ls_{}.npz'.format(args.save_path, args.centering, args.loss_mode, args.lr, args.weight_decay, args.label_scale), valid_acc = valid_acc, test_acc = test_acc, test_predictions = test_predictions)        
            
            
    if args.run_krr:
        output_file.write("\nRunning KRR results\n")
        output_file.flush()
        print("Running KRR results")
        
        _, _, kernel_fn = kernels.DCConvNetKernel(depth = 3, width = 1024, num_classes = n_classes)
        KERNEL_FN = functools.partial(kernel_fn, get=('nngp', 'ntk'))
        
        kernel_batch_size = 25 if X_sup.shape[0]%25 == 0 else 20
        # KERNEL_FN = nt.utils.batch.batch(KERNEL_FN, batch_size=kernel_batch_size)
        KERNEL_FN = nt.batch(KERNEL_FN, batch_size=kernel_batch_size)
        
        X_sup_reordered = np.transpose(X_sup, [0,2,3,1])
            
        X_valid_reordered = np.transpose(X_valid.numpy(), [0,2,3,1])
        X_test_reordered = np.transpose(X_test.numpy(), [0,2,3,1])
        
        K_zz = KERNEL_FN(X_sup_reordered, X_sup_reordered)
        
        K_zz_nngp = K_zz.nngp + (jit * np.eye(K_zz.nngp.shape[0]) * np.trace(K_zz.nngp)/K_zz.nngp.shape[0])
        
        K_zz_ntk = K_zz.ntk + (jit * np.eye(K_zz.ntk.shape[0]) * np.trace(K_zz.ntk)/K_zz.ntk.shape[0])
        
        for eval_set, y_eval, eval_set_name in zip([X_valid_reordered, X_test_reordered], [y_valid.numpy(), y_test.numpy()], ['valid', 'test']):
            if eval_set.shape[0] % kernel_batch_size != 0:
                K_xz0 = KERNEL_FN(eval_set[:-(eval_set.shape[0] % kernel_batch_size)], X_sup_reordered)
                K_xz1 = KERNEL_FN(eval_set[-(eval_set.shape[0] % kernel_batch_size):], X_sup_reordered)
                K_xz_nngp = np.concatenate([K_xz0.nngp, K_xz1.nngp], 0)
                K_xz_ntk = np.concatenate([K_xz0.ntk, K_xz1.ntk], 0)
            else:
                K_xz = KERNEL_FN(eval_set, X_sup_reordered)
                K_xz_nngp = K_xz.nngp
                K_xz_ntk = K_xz.ntk
            preds_nngp = np.array(K_xz_nngp) @ np.linalg.solve(np.array(K_zz_nngp), y_sup)
            acc_nngp = np.mean(np.argmax(preds_nngp, 1) == y_eval)
            
            preds_ntk = np.array(K_xz_ntk) @ np.linalg.solve(np.array(K_zz_ntk), y_sup)
            acc_ntk = np.mean(np.argmax(preds_ntk, 1) == y_eval)
            file_print('KRR results on {} set: NNGP: {}, NTK: {}\n'.format(eval_set_name, acc_nngp, acc_ntk))
            
            if args.no_kernel_save:
                np.savez('{}/{}_kernels_{}.npz'.format(args.save_path, args.epoch, eval_set_name), K_zz_nngp = K_zz.nngp, K_xz_nngp = K_xz_nngp, K_zz_ntk = K_zz.ntk, K_xz_ntk = K_xz_ntk, acc_nngp = acc_nngp, acc_ntk = acc_ntk)        
            else:
                np.savez('{}/{}_kernels_{}.npz'.format(args.save_path, args.epoch, eval_set_name), acc_nngp = acc_nngp, acc_ntk = acc_ntk)        
                   
    output_file.close()
                                
                                


if __name__ == '__main__':
    main()