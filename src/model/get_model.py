import torch
from .model import LeNet, ResNet, BasicBlock
from torchvision import models

"""
Functions:
    get_model
"""

def get_model(config):
    """
    Return model
    """
    print('Creating model...')

    config_model = config['model']
    model_name = config_model['type']
    init_method = config_model['init_method']
    
    config_data = config['data']
    in_channels = config_data['image_channels']
    num_classes = config_data['num_classes']
    
    if model_name == 'resnet18':
        num_blocks = config_model['num_blocks']
        model = ResNet(BasicBlock, num_blocks, in_channels, num_classes, init_method)
    elif model_name == "LeNet":
        model = LeNet()
    elif model_name == 'vgg11':
        if config['model']['pre_trained']:
            model = models.vgg11(weights='DEFAULT')
        else:
            model = models.vgg11(weights=None)
    else:
        raise NotImplementedError(f'Model {model_name} not implemented')
    
    print(f'Model {model_name} created')
    
    return model


