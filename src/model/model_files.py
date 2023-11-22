# Method to save model
import torch

"""
Functions:
    save_model
    read_model
"""

def save_model(model, config):
    """
    Save model
    :param model: model to save
    :param config: config dict
    :return: None
    """

    print('Saving model...')

    # set model to eval mode
    model.eval()

    # check if the path is valid
    if not config['model']['save_path'].endswith('.pth'):
        raise ValueError('Model save path must end with .pth')

    # save model
    torch.save(model.state_dict(), config['model']['save_path'])

    print('Model saved')

def read_model(model, config):
    """
    Read model's state dictionary from a file and load it into the model.
    :param model: model to read
    :param config: config dict containing the model's save path
    :return: None
    """

    # Ensure the path to the model's saved state dict is provided
    if 'model' in config and 'save_path' in config['model']:
        model_save_path = config['model']['save_path']

        # Load the model's state dict from the specified path
        model.load_state_dict(torch.load(model_save_path))

        # Set the model to evaluation mode
        model.eval()

        print('Model read successfully from', model_save_path)
    else:
        print('Model save path not found in the config.')
