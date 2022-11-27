# Efficient Dataset Distillation using Random Feature Approximation

Code for the NeurIPS paper ["Efficient Dataset Distillation using Random Feature Approximation"](https://arxiv.org/abs/2210.12067)

Contact: [Noel Loo](loo@mit.edu)

# Abstract
Dataset distillation compresses large datasets into smaller synthetic coresets which retain performance with the aim of reducing the storage and computational burden of processing the entire dataset. 
Today's best-performing algorithm, *Kernel Inducing Points* (KIP), which makes use of the correspondence between infinite-width neural networks and kernel-ridge regression, is prohibitively slow due to the exact computation of the neural tangent kernel matrix, scaling O(|S|2), with |S| being the coreset size. To improve this, we propose a novel algorithm that uses a random feature approximation (RFA) of the Neural Network Gaussian Process (NNGP) kernel, which reduces the kernel matrix computation to O(|S|). Our algorithm provides at least a 100-fold speedup over KIP and can run on a single GPU. Our new method, termed an RFA Distillation (RFAD), performs competitively with KIP and other dataset condensation algorithms in accuracy over a range of large-scale datasets, both in kernel regression and finite-width network training. We demonstrate the effectiveness of our approach on tasks involving model interpretability and privacy preservation.

# Example usage
To run generate a distilled set on cifar10, 10 samples per class, platt loss with label learning, for example:

```python3 run_distillation.py --dataset cifar10 --save_path path/to/directory/ --samples_per_class 10 --platt --learn_labels ```

This does not automatically evaluate the dataset on the test set.

To evaluate a distilled set with NNGP/NTK kernel ridge regression with an already made distilled dataset on all datasets except celebA:
```python3 eval_distilled_set.py --dataset fashion --save_path path/to/directory --run_krr```

To evaluate a distilled set with a finite network trained with SGD on mnist, with an already made distilled dataset:

```python3 eval_distilled_set.py --dataset mnist --save_path path/to/directory --run_finite --lr 1e-3 --weight_decay 1e-3 --label_scale 8` --centering ```

utils.py contains the best hyperparameters for finite network training

To use the empirical NNGP for inference on fashion-mnist:
```python3 run_network_parameter_analysis.py --dataset fashion --save_path path/to/directory```

To use the empirical NNGP for inference on fashion-mnist:
```python3 run_network_parameter_analysis.py --dataset fashion --save_path path/to/directory```

To run the time profiling experiment for model counts of 1,2,4,8, for samples per class in the coreset of 1,5,10,20,50:
```python3 run_time_profile_exp.py --dataset cifar10 --n_models 1 2 4 8 --samples_per_class 1 5 10 20 50```

To run corruption experiments on CelebA with corruption 0.8:
```python3 run_distillation.py --dataset celeba --save_path path/to/directory/ --samples_per_class 1 --platt --n_batches 1 --init_strategy noise --corruption 0.8```
To run CelebA experiments, make sure you are on the latest version of PyTorch, as older version have a bug where the test/train splis are incorrect.

To evaluate with NNGP KRR on CelebA:
```python3 eval_distilled_set_batched.py --dataset celeba --save_path path/to/directory --run_krr```

We additionally include some distilled dataset for cifar10 with 1,10, or 50 samples per class in ./distilled_images_final/cifar10 in the files 'best.npz'


# Requirements
- pytorch
- neural-tangents
- torch_optimizer
- sklearn, matplotlib, numpy, scipy
Note that some versions of pytorch have incorrect test/train splits for CelebA