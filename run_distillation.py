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
import coresets
from torchvision import datasets,transforms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--jit', type=float, default = 5e-3)
    parser.add_argument('--save_path', type = str)
    parser.add_argument('--samples_per_class', type = int)
    parser.add_argument('--init_strategy', type=str, default = 'random')
    parser.add_argument('--learn_labels', action='store_true')
    parser.add_argument('--platt', action='store_true')
    parser.add_argument('--coreset', action='store_true')
    parser.add_argument('--n_models', type=int, default = 8)
    parser.add_argument('--ga_steps', type=int, default = 1)
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--corruption', type=float, default = 0)
    parser.add_argument('--n_batches', type = int, default = 4)

    args = parser.parse_args()
    
    transform_fn = None
    from_loader = False
    whitening_mat = None
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
        ds_train = datasets.CelebA('./data/', split = 'train', download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([64, 64]), transforms.Normalize((0.5064, 0.4258, 0.3832), (0.3093, 0.2890, 0.2883))]), target_type = 'attr',
                          target_transform = transforms.Lambda(lambda x: x[20]))
        train_loader = torch.utils.data.DataLoader(ds_train,
                                          batch_size=1280,
                                          shuffle=True,
                                          num_workers=8)
        
        ds_valid = datasets.CelebA('./data/', split = 'valid', download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([64, 64]), transforms.Normalize((0.5064, 0.4258, 0.3832), (0.3093, 0.2890, 0.2883))]), target_type = 'attr',
                          target_transform = transforms.Lambda(lambda x: x[20]))
        
        valid_set = next(iter(torch.utils.data.DataLoader(ds_valid,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=8)))
        
        X_valid = valid_set[0]
        y_valid = valid_set[1]
        
        X_train = train_loader
        y_train = None
        from_loader = True
        
    else:
        print("unrecognized dataset: {}".format(args.dataset))
        sys.exit()
        
    if args.dataset != 'celeba':
        X_init = coresets.make_coreset(X_train, y_train, args.samples_per_class, n_classes, args.init_strategy, seed = args.seed)
    else:
        batch = next(iter(X_train))
        X_init = coresets.make_coreset(batch[0], batch[1], args.samples_per_class, n_classes, args.init_strategy, seed = args.seed)
    
    if args.dataset != 'celeba':
        np.random.seed(args.seed)
        valid_indices = []
        for c in range(n_classes):
            class_indices = np.where(y_train == c)[0]
            valid_indices.append(class_indices[np.random.choice(len(class_indices), 500 if n_classes == 10 else 100)])

        valid_indices = np.concatenate(valid_indices)
        X_valid = X_train[valid_indices]
        y_valid = y_train[valid_indices]
    
    model_class = partial(models.ConvNet_wide, n_channels, net_norm = 'none', im_size=(im_size,im_size), k = 2, chopped_head = True)
    scheduler = [(0, args.n_models, 1)]
    
    n_iters = 100000 if not args.coreset else 1
    
    distillation.distill_dataset(X_train, y_train,
                                       model_class, args.lr, 8, args.n_batches, iters = n_iters,
                                       ga_steps = args.ga_steps, platt = args.platt, 
                                       schedule = scheduler, save_location = args.save_path,
                                      samples_per_class = args.samples_per_class, n_classes = n_classes, learn_labels = args.learn_labels,
                                      batch_size = 1280, X_valid = X_valid, y_valid = y_valid,
                                      n_channels = n_channels, im_size = im_size, X_init = X_init, jit = args.jit, seed = args.seed, corruption = args.corruption, whitening_mat = whitening_mat, from_loader = from_loader)