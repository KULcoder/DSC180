import torch
from .model import ResNet, BasicBlock

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
    num_blocks = config_model['num_blocks']
    config_data = config['data']
    in_channels = config_data['image_channels']
    num_classes = config_data['num_classes']
    
    if model_name == 'resnet18':
        model = ResNet(BasicBlock, num_blocks, in_channels, num_classes)
    else:
        raise NotImplementedError(f'Model {model_name} not implemented')
    
    print(f'Model {model_name} created')
    
    return model


