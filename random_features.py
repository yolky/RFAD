import torch
import numpy as np
import math

def get_random_features(X_train, model_class, n_models, n_features_per_model, batch_size = 300, fixed_seed = None, other_datasets = [], featurewise_normalize = False, elementwise_normalize = False, device = 'cuda:0', return_models = False):
    created_feature_vec = False
    
    models = []
    
    for m in range(n_models):
        if fixed_seed is not None:
            torch.manual_seed(fixed_seed[m])
        model = model_class(n_random_features = n_features_per_model)

        model.to(device)
        model.eval()
        
        if return_models:
            models.append(model)

        for i in range(math.ceil(X_train.shape[0]/batch_size)):
            with torch.no_grad():
                out = model(torch.Tensor(X_train[batch_size * i: batch_size * (i+1)]).to(device))
                
                if not created_feature_vec:
                    n_features_per_model = out.shape[1]
                    X_train_features = np.zeros([X_train.shape[0], n_models * n_features_per_model], dtype = np.float32)
                    other_features = [np.zeros([ds.shape[0], n_models * n_features_per_model], dtype = np.float32) for ds in other_datasets]
                    created_feature_vec = True
                
                X_train_features[batch_size * i: batch_size * (i+1), n_features_per_model * m: n_features_per_model * (m+1)] = ((out.detach().cpu().numpy())/np.sqrt(n_models * n_features_per_model))
                                
        for ds_index in range(len(other_datasets)):
            dataset = other_datasets[ds_index]
            for i in range(math.ceil(dataset.shape[0]/batch_size)):
                with torch.no_grad():
                    out = model(torch.Tensor(dataset[batch_size * i: batch_size * (i+1)]).to(device))
                    other_features[ds_index][batch_size * i: batch_size * (i+1), n_features_per_model * m: n_features_per_model * (m+1)] = ((out.detach().cpu().numpy())/np.sqrt(n_models * n_features_per_model))
                        
    if(featurewise_normalize):
        feature_means = np.mean(X_train_features, axis = 0, keepdims = True)
        X_train_features = X_train_features - feature_means
        for j in range(len(other_features)):
            other_features[j] = other_features[j] - feature_means
    
    if elementwise_normalize:
        X_train_features = X_train_features - np.mean(X_train_features, axis = 1, keepdims = True)
        for j in range(len(other_features)):
            other_features[j] = other_features[j] - np.mean(other_features[j], axis = 1, keepdims = True)
        
    if elementwise_normalize:
        bongo_mean = np.mean(X_train_features)
        X_train_features = X_train_features - bongo_mean
        for other_feature in other_features:
            other_feature = other_feature - bongo_mean
    
    if return_models:
            return X_train_features, other_features, models
        
    return X_train_features, other_features