"""
Direct script to run an experiment.

Design: 

Ways to run:

python3 experiments/train_test.py config_file

Example

python3 experiments/train_test.py vgg11_config_py.json

"""

import sys
import os
# temporarily append the upper dir to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time

from src.model.get_model import get_model
from src.experiment.Experiment import Experiment
from src.experiment.utils import save_log
from src.model.model_files import save_model

if __name__ == '__main__':

    config_file = sys.argv[1]

    # read the config
    config_path = os.path.join("config", config_file)
    with open(config_path) as json_file:
        config = json.load(json_file)

    exp = Experiment(config = config)
    exp.run()

    # Save the model
    if config['model']['save_model']:
        model = exp.get_model()
        save_name = config['model']['type']+"_"+config['experiment_name']+"_"+str(round(time.time()))+".pth"
        save_path = os.path.join(config["model"]["save_path"], save_name)
        config['model']['save_path'] = save_path
        save_model(model, config)

    # Log the file
    log_name = config['experiment_name']+"_"+str(round(time.time()))+".json"

    os.makedirs(config["training"]['log_path'])
    log_path = os.path.join(config["training"]['log_path'], log_name)
    save_log(exp.get_log(), log_path)

    
