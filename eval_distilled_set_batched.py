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

from torchvision import datasets,transforms

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_path', type = str)
    parser.add_argument('--epoch', type = str, default = 'best')
    parser.add_argument('--run_krr', action='store_true')
    parser.add_argument('--run_finite', action='store_true')
    parser.add_argument('--valid_seed', type = int, default = 0)
    
    parser.add_argument('--centering', action='store_true')

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
        
    elif args.dataset == 'celeba':
        im_size = 64
        n_channels = 3
        n_classes = 2
        ds_train = datasets.CelebA('./data/', split = 'test', download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([64, 64]), transforms.Normalize((0.5064, 0.4258, 0.3832), (0.3093, 0.2890, 0.2883))]), target_type = 'attr',
                          target_transform = transforms.Lambda(lambda x: x[20]))
        test_loader = torch.utils.data.DataLoader(ds_train,
                                          batch_size=1280,
                                          shuffle=True,
                                          num_workers=8)

    else:
        print("unrecognized dataset: {}".format(args.dataset))
        sys.exit()
    
    print("Evaluating at path {} on dataset {}".format(args.save_path, args.dataset))
    
    support_set = np.load('{}/{}.npz'.format(args.save_path, args.epoch))
    X_sup = support_set['images']
    y_sup = support_set['labels']
    # print(y_sup)
    # print(support_set['k'])
    jit = support_set.get('jit', 5e-3)

    
    output_file = open('{}/eval_results_{}.txt'.format(args.save_path, datetime.datetime.now()) ,'a')
    file_print = partial(double_print, output_file = output_file)

            
    if args.run_krr:
        output_file.write("\nRunning KRR results\n")
        output_file.flush()
        print("Running KRR results")
        
        _, _, kernel_fn = kernels.DCConvNetKernel(depth = 3, width = 1024, num_classes = n_classes)
        KERNEL_FN = functools.partial(kernel_fn, get=('nngp', 'ntk'))
        
        kernel_batch_size = 25 if X_sup.shape[0]%25 == 0 else 20
        KERNEL_FN = nt.utils.batch.batch(KERNEL_FN, batch_size=kernel_batch_size)
        
        X_sup_reordered = np.transpose(X_sup, [0,2,3,1])
        
        K_zz = KERNEL_FN(X_sup_reordered, X_sup_reordered)
        
        K_zz_nngp = K_zz.nngp + (jit * np.eye(K_zz.nngp.shape[0]) * np.trace(K_zz.nngp)/K_zz.nngp.shape[0])
        
        K_zz_ntk = K_zz.ntk + (jit * np.eye(K_zz.ntk.shape[0]) * np.trace(K_zz.ntk)/K_zz.ntk.shape[0])
        total_count = 0
        total_correct_nngp = 0
        total_correct_ntk = 0
        batch = 0
        for test_batch in test_loader:
        # for eval_set, y_eval, eval_set_name in zip([X_valid_reordered, X_test_reordered], [y_valid.numpy(), y_test.numpy()], ['valid', 'test']):
            eval_set = np.transpose(test_batch[0].numpy(), [0,2,3,1])
            y_eval = test_batch[1].numpy()
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
            
            
            print('batch {}: acc_nngp: {}, acc_ntk: {}'.format(batch + 1, acc_nngp, acc_ntk))
            
            total_count += len(y_eval)
            
            total_correct_nngp += np.sum(np.argmax(preds_nngp, 1) == y_eval)
            
            total_correct_ntk += np.sum(np.argmax(preds_ntk, 1) == y_eval)
            batch += 1
#             file_print('KRR results on {} set: NNGP: {}, NTK: {}\n'.format(eval_set_name, acc_nngp, acc_ntk))
            
#             np.savez('{}/{}_kernels_{}.npz'.format(args.save_path, args.epoch, eval_set_name), K_zz_nngp = K_zz.nngp, K_xz_nngp = K_xz_nngp, K_zz_ntk = K_zz.ntk, K_xz_ntk = K_xz_ntk, acc_nngp = acc_nngp, acc_ntk = acc_ntk)        
                   
    acc_nngp = total_correct_nngp/total_count
    acc_ntk = total_correct_ntk/total_count
    file_print('KRR results on test set: NNGP: {}, NTK: {}\n'.format(acc_nngp, acc_ntk))
    
    np.savez('{}/{}_kernels_{}.npz'.format(args.save_path, args.epoch, 'test'), acc_nngp = acc_nngp, acc_ntk = acc_ntk)        

    output_file.close()
                                
                                


if __name__ == '__main__':
    main()