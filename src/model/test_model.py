import torch 
import torch.nn as nn

from .evaluate_model import calculate_acc
from .utils import one_hot_encode

"""
Functions:
    test_model
"""


def test_model(model, test_dataloader, config):
    """
    Test model
    :param model: model to test
    :param test_dataloader: test dataloader
    :param config: config dict
    :return: testing loss, testing accuracy
    """

    print('Testing model...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # set model to eval mode
    model.eval()

    testing_loss = 0.0
    testing_correct = 0.0

    criterion_type = config['training']['criterion']
    if criterion_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f'Criterion {criterion} not implemented')
    
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        if type(criterion) == nn.MSELoss:
            one_hot_labels = one_hot_encode(labels, config['data']['num_classes']).to(device)
            loss = criterion(outputs, one_hot_labels)
        else:
            loss = criterion(outputs, labels)

        correct, accuracy = calculate_acc(outputs, labels)

        testing_correct += correct
        testing_loss += loss.detach().cpu().item()

    # record testing loss and accuracy
    testing_loss /= len(test_dataloader)
    testing_acc = 100 * testing_correct / len(test_dataloader.dataset)

    print(f'Testing loss: {testing_loss:.4f}, Testing accuracy: {testing_acc:.4f}')

    return testing_loss, testing_acc
