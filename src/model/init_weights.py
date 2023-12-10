import torch.nn.init as init

def init_weights(config, model):
    """
    Intilize the weight of model based on:
    - type of initilization
    - type of model
    """

    if config['model']['init_method'] == "zero":
        return zero_init(config, model)
    elif config['model']['init_method'] == "normal":
        return normal_init(config, model)
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
    else:
        raise NotImplementedError(f'Model {config["model"]["type"]} normal init not implemented')

    return model

