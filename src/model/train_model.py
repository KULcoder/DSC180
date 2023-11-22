import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from .evaluate_model import calculate_acc
from .utils import one_hot_encode

"""
Functions:
    train_model
"""

def train_model(model, train_dataloader, val_dataloader, config):
    """
    Train model
    :param model: model to train
    :param train_dataloader: train dataloader
    :param val_dataloader: validation dataloader
    :param config: config dict
    :return: trained_model, training_losses, training_accuracies, validation_losses, validation_accuracies
    """

    print('Training model...')

    # device for cuda, mps, or cpu
    # notice this implementation only supports single GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device {device}')

    model = model.to(device)
    
    # for ynamic display loss and accuracy
    live_losses = []
    live_accs = []

    epochs = config['training']['epochs']
    epoch_describer = tqdm(range(epochs), desc=f"Train ", ncols=100)

    training_losses = []
    training_accs = []
    validation_losses = []
    validation_accs = []

    # define loss function and optimizer
    config_train = config['training']
    criterion_type = config_train['criterion']
    if criterion_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f'Criterion {criterion} not implemented')
    
    config_optimizer = config['optimizer']
    optimizer_type = config_optimizer['type']
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config_optimizer['lr'], 
            weight_decay=config_optimizer['weight_decay']
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config_optimizer['lr'], 
            momentum=config_optimizer['momentum'], 
            weight_decay=config_optimizer['weight_decay'],
            nesterov=config_optimizer['nestrov']
        )
    else:
        raise NotImplementedError(f'Optimizer {optimizer} not implemented')
    
    for epoch in epoch_describer:

        # TRAINING
        # set model to train mode
        model.train()

        training_loss = 0.0
        training_correct = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            if type(criterion) == nn.MSELoss:
                one_hot_labels = one_hot_encode(labels, config['data']['num_classes']).to(device)
                loss = criterion(outputs, one_hot_labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct, accuracy = calculate_acc(outputs, labels)

            training_correct += correct
            training_loss += loss.detach().cpu().item()

            live_losses.append(loss.detach().cpu().item())
            live_accs.append(accuracy)
            if len(live_losses) > 100:
                live_losses.pop(0)
                live_accs.pop(0)
            epoch_describer.\
                set_description(f"Train (loss={np.mean(live_losses):.3f}, acc={np.mean(live_accs):.3f})")
            
        # record training loss and accuracy
        training_losses.append(training_loss / len(train_dataloader))
        training_accs.append(training_correct / len(train_dataloader.dataset))

        # VALIDATION
        # set model to eval mode
        model.eval()

        valid_loss = 0.0
        valid_correct = 0.0

        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if type(criterion) == nn.MSELoss:
                one_hot_labels = one_hot_encode(labels, config['data']['num_classes']).to(device)
                loss = criterion(outputs, one_hot_labels)
            else:
                loss = criterion(outputs, labels)

            correct, accuracy = calculate_acc(outputs, labels)

            valid_correct += correct
            valid_loss += loss.detach().cpu().item()

        # record validation loss and accuracy
        validation_losses.append(valid_loss / len(val_dataloader))
        validation_accs.append(valid_correct / len(val_dataloader.dataset))

    return model, training_losses, training_accs, validation_losses, validation_accs


