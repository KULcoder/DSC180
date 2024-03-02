import torch.nn.init as init
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal

from scipy.linalg import eigvals
import numpy as np
import os

def init_weights(config, model):
    """
    Intilize the weight of model based on:
    - type of initilization
    - type of model
    """

    if config['model']['init_method'] == 'default':
        return model
    elif config['model']['init_method'] == "zero":
        return zero_init(config, model)
    elif config['model']['init_method'] == "normal":
        return normal_init(config, model)
    elif config['model']['init_method'] == 'agop':
        return agop_init(config, model)
    elif config['model']['init_method'] == 'nfm':
        return nfm_init(config, model)
    else:
        raise NotImplementedError(f'Model {config["model"]["type"]} not implemented')
    

def zero_init(config, model):
    """
    Intilize the weight of model with zero
    """
    if config['model']['type'] == 'resnet18':
        init.constant_(model.conv1.weight, 0)
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                init.constant_(block.conv1.weight, 0)
                init.constant_(block.conv2.weight, 0)
                if block.shortcut:
                    init.constant_(block.shortcut[0].weight, 0)
    elif config['model']['type'] == "LeNet":
        init.constant_(model.conv1.weight, 0)
        init.constant_(model.conv2.weight, 0)
        init.constant_(model.fc1.weight, 0)
        init.constant_(model.fc2.weight, 0)
        init.constant_(model.fc3.weight, 0)
    elif config['model']['type'] == "vgg11":
        for name, module in model.features.named_children():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                
    else:
        raise NotImplementedError(f'Model {config["model"]["type"]} zero init not implemented')
    
    return model

def normal_init(config, model):
    """
    Intilize the weight of model with normal distribution
    """
    init_mean = config['model']['init_mean']
    init_std = config['model']['init_std']

    if config['model']['type'] == 'resnet18':
        init.normal_(model.conv1.weight, init_mean, init_std)
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                init.normal_(block.conv1.weight, init_mean, init_std)
                init.normal_(block.conv2.weight, init_mean, init_std)
                if block.shortcut:
                    init.normal_(block.shortcut[0].weight, init_mean, init_std)
    elif config['model']['type'] == "LeNet":
        init.normal_(model.conv1.weight, init_mean, init_std)
        init.normal_(model.conv2.weight, init_mean, init_std)
        init.normal_(model.fc1.weight, init_mean, init_std)
        init.normal_(model.fc2.weight, init_mean, init_std)
        init.normal_(model.fc3.weight, init_mean, init_std)
    elif config['model']['type'] == "vgg11":
        for name, module in model.features.named_children():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight,
                    mean = init_mean,
                    std = init_std
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    else:
        raise NotImplementedError(f'Model {config["model"]["type"]} normal init not implemented')

    return model


def init_conv_with_cov(conv_layer, cov_matrix):
    """
    Given the weight of a conv layer can covariance matrix, created a init with that.
    In place operation
    """

    out_channel, in_channel, kernel_size, kernel_size = conv_layer.weight.data.shape
    means = torch.zeros(in_channel * kernel_size * kernel_size)
    distribution = MultivariateNormal(means, cov_matrix)
    # total out_channel amount of samples
    new_weights = distribution.sample((out_channel, )).reshape(out_channel, in_channel, kernel_size, kernel_size)
    new_weights.requires_grad = True
    conv_layer.weight = nn.Parameter(new_weights)

def transform_matrix(A):
    # Making positive definite
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    min_eigenvalue = min(eigenvalues)
    if min_eigenvalue <= 0:
        positive_eigenvalues = eigenvalues + (np.abs(min_eigenvalue) + 1e-6)
        A = eigenvectors @ np.diag(positive_eigenvalues) @ eigenvectors.T
    
    # Normalizing by dividing by the trace
    trace_A = np.trace(A)
    if trace_A > 0:
        A = A / trace_A

    return A 

def agop_init(config, model):
    """
    Initialize the weight of convolution layers with agop as covariance.
    ONLY FOR VGG11
    """
    
    conv_layer_index = 0
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            # load the agop
            agop_path = os.path.join(config['model']['agop_path'], f"layer_{conv_layer_index}.csv")
            agop = np.loadtxt(agop_path, delimiter=',')
            agop = transform_matrix(agop)
            agop = torch.from_numpy(agop).float()
            init_conv_with_cov(module, agop)
            conv_layer_index += 1

    return model
            

def nfm_init(config, model):
    """
    Initialize the weight of convolution layers with nfm as covariance matrix.
    ONLY FOR VGG11
    """
    conv_layer_index = 0
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            # load the nfm
            nfm_path = os.path.join(config['model']['nfm_path'], f"layer_{conv_layer_index}.csv")
            nfm = np.loadtxt(nfm_path, delimiter=',')
            nfm = transform_matrix(nfm)
            nfm = torch.from_numpy(nfm).float()
            init_conv_with_cov(module, nfm)
            conv_layer_index += 1

    return model

