import numpy as np

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def double_print(text, output_file = None, end = '\n'):
    print(text, end = end)
    if not output_file is None:
        output_file.write(str(text) + end)
        output_file.flush()
        
def get_best_finite_hypers(ds, ss, use_label_scale = True, use_instance = False, from_instance = False):
    #hyperparameters for finite networks trained with gradient descent
    if use_instance:
        if not from_instance:
            if ss == 1:
                lr = 1e-4
                ls = 8
                wd = 0
            elif ss == 10:
                lr = 1e-3
                ls = 8
                wd = 0
            elif ss == 50:
                lr = 1e-3
                ls = 2
                wd = 0
        else:
            if ss == 1:
                lr = 1e-3
                ls = 16
                wd = 0
            elif ss == 10:
                lr = 1e-2
                ls = 8
                wd = 0
            elif ss == 50:
                lr = 1e-3
                ls = 2
                wd = 0
    elif from_instance:
        if ss == 1:
            lr = 1e-4
            ls = 16
            wd = 0
        elif ss == 10:
            lr = 1e-4
            ls = 8
            wd = 0
        elif ss == 50:
            lr = 1e-2
            ls = 8
            wd = 0
    
    elif use_label_scale:
        if ss == 1:
            if ds == 'mnist':
                lr = 1e-3
                ls = 8
                wd = 1e-3
            elif ds == 'fashion':
                lr = 1e-3
                ls = 8
                wd = 0
            elif ds == 'svhn':
                lr = 1e-3
                ls = 16
                wd = 0
            elif ds == 'cifar10':
                lr = 1e-3
                ls = 8
                wd = 0
            elif ds == 'cifar100':
                lr = 1e-1
                ls = 8
                wd = 0
        elif ss == 10:
            if ds == 'mnist':
                lr = 1e-2
                ls = 8
                wd = 1e-3
            elif ds == 'fashion':
                lr = 1e-2
                ls = 8
                wd = 1e-3
            elif ds == 'svhn':
                lr = 1e-2
                ls = 8
                wd = 0
            elif ds == 'cifar10':
                lr = 1e-3
                ls = 2
                wd = 0
            elif ds == 'cifar100':
                lr = 1e-1
                ls = 2
                wd = 0
        elif ss == 50:
            if ds == 'mnist':
                lr = 1e-2
                ls = 8
                wd = 0
            elif ds == 'fashion':
                lr = 1e-3
                ls = 2
                wd = 0
            elif ds == 'svhn':
                lr = 1e-2
                ls = 2
                wd = 0
            elif ds == 'cifar10':
                lr = 1e-2
                ls = 2
                wd = 0
    else:
        ls = 1
        
        if ss == 1:
            if ds == 'mnist':
                lr = 1e-5
                wd = 0
            elif ds == 'fashion':
                lr = 1e-3
                wd = 0
            elif ds == 'svhn':
                lr = 1e-4
                wd = 0
            elif ds == 'cifar10':
                lr = 1e-4
                wd = 0
            elif ds == 'cifar100':
                lr = 1e-2
                wd = 0
        elif ss == 10:
            if ds == 'mnist':
                lr = 1e-4
                wd = 1e-3
            elif ds == 'fashion':
                lr = 1e-4
                wd = 1e-3
            elif ds == 'svhn':
                lr = 1e-3
                wd = 0
            elif ds == 'cifar10':
                lr = 1e-4
                wd = 0
            elif ds == 'cifar100':
                lr = 1e-2
                wd = 0
        elif ss == 50:
            if ds == 'mnist':
                lr = 1e-3
                wd = 0
            elif ds == 'fashion':
                lr = 1e-3
                wd = 0
            elif ds == 'svhn':
                lr = 1e-3
                wd = 0
            elif ds == 'cifar10':
                lr = 1e-3
                wd = 0
    
    return lr, ls, wd