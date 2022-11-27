import neural_tangents as nt
from neural_tangents import stax
import numpy as np
import functools

def DCConvNetKernel( 
        depth,
        width,
        W_std = np.sqrt(2), 
        b_std = 0.1,
        num_classes = 10,
        parameterization = 'ntk',
        activation = 'relu'):
    """Returns neural_tangents.stax fully convolutional network."""
    activation_fn = stax.Relu()
    conv0 = functools.partial(
            stax.Conv,
            W_std=W_std,
            b_std=b_std,
            padding='SAME',
            parameterization=parameterization)
    conv = functools.partial(
            stax.Conv,
            W_std=W_std,
            b_std=b_std,
            padding='SAME',
            parameterization=parameterization)
    layers = []
    for d in range(depth):
        layers += [conv(width, (3,3)), activation_fn, stax.AvgPool((2,2), strides=(2, 2))]
    layers += [stax.Flatten(), stax.Dense(num_classes, W_std=W_std, b_std=b_std,parameterization=parameterization)]
    
    return stax.serial(*layers)