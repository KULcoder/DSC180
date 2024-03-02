"""
Direct script to run an experiment.

Design: 

Ways to run:

python3 experiments/train_test.py experiment_name

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

if __name__ == '__main__':

    exp_name = sys.argv[1]
    args = sys.argv[2:]

    # read the config
    config_path = os.path.join("config", "vgg11_config_py.json")
    with open(config_path) as json_file:
        config = json.load(json_file)

    exp = Experiment(config = config)
    exp.run()

    log_name = config['experiment_name']+"_"+str(round(time.time()))+".json"

    os.makedirs(config["training"]['log_path'])
    log_path = os.path.join(config["training"]['log_path'], log_name)
    save_log(exp.get_log(), log_path)

    
