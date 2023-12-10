from src.data.get_dataloader import get_dataloaders
from src.model.get_model import get_model
from src.visualization.acc_loss import plot_acc_loss
from src.model.init_weights import init_weights
from src.experiment.utils import one_hot_encode

import os
import json
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class Experiment(object):
    """
    The Experiment Object which
    - store dataloader, models, logs
    - execute training & testing
    - can use other methods to log results to files
    """
    def __init__(self, config):
        # Prepare config
        if type(config) == str:
            try:
                with open(config) as json_file:
                    self.__config = json.load(json_file)
            except Exception as e:
                print("Error when loading config")
                print("Error:", e)
                return
        elif type(config) == dict:
            self.__config = config

        # prepare device
        if torch.cuda.is_available():
            self.__device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.__device = torch.device('mps')
        else:
            self.__device = torch.device('cpu')
        print(f'Using device {self.__device}')
        # prepare dataloader
        self.__train_loader, \
        self.__val_loader, \
        self.__test_loader = get_dataloaders(self.__config)
        # prepare model
        self.__model = get_model(self.__config)
        self.__model = init_weights(self.__config, self.__model)
        self.__model.to(self.__device).float()
        # prepare criterion
        if self.__config['training']['criterion'] == "cross_entropy":
            self.__criterion = nn.CrossEntropyLoss()
        elif self.__config['training']['criterion'] == "MSE":
            self.__criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f'Criterion {config["training"]["criterion"]} not implemented')
        # prepare optimizer
        config_optimizer = self.__config['optimizer']
        if config_optimizer['type'] == 'adam':
            self.__optimizer = torch.optim.Adam(
                self.__model.parameters(),
                lr = config_optimizer['lr'],
                weight_decay = config_optimizer['weight_decay']
            )
        elif config_optimizer['type'] == 'sgd':
            self.__optimizer = torch.optim.SGD(
                self.__model.parameters(), 
                lr=config_optimizer['lr'], 
                momentum=config_optimizer['momentum'], 
                weight_decay=config_optimizer['weight_decay'],
                nesterov=config_optimizer['nestrov']
            )
        else:
            raise NotImplementedError(f'Optimizer {config_optimizer["type"]} not implemented')
        # TODO: prepare LR scheduler

        # store the logs
        self.__train_losses = []
        self.__val_losses = []
        self.__train_accs = []
        self.__val_accs = []

        # util parameters
        self.__current_epoch = 0
        self.__epochs = self.__config['training']['epochs']

    def run(self, run_epochs=None):
        print('\t\trunning experiment')
        start_epoch = self.__current_epoch
        if run_epochs != None:
            end_epoch = start_epoch + run_epochs
        else:
            end_epoch = self.__epochs
            if start_epoch >= end_epoch:
                print("Running Epoch over Config Epoch, specify run_epochs for more epochs")
                return

        # here we implement the tqdm live progress bar
        epoch_describer = tqdm(range(start_epoch, end_epoch), desc=f"Train", ncols=100)
        live_stats = {
            "describer": epoch_describer,
            "loss": [],
            "acc": []
        }
        
        for epoch in epoch_describer:
            self.__current_epoch = epoch
            # train
            train_loss, train_acc = self.__train(live_stats)
            self.__train_losses.append(train_loss)
            self.__train_accs.append(train_acc)
            # validation
            val_loss, val_acc = self.__test(validation=True)
            self.__val_losses.append(val_loss)
            self.__val_accs.append(val_acc)

            # TODO: lr_schedular

            epoch_describer.\
                set_description(f"Train (loss={np.mean(live_stats['loss']):.3f}, acc={np.mean(live_stats['acc']):.3f})")

        test_loss, test_acc = self.__test(validation=False)
        
        return self.__train_losses, \
            self.__train_accs, \
            self.__val_losses, \
            self.__val_accs, \
            test_loss, \
            test_acc

    def __compute_loss_accuracy(self, inputs, labels):
        logits = self.__model(inputs)
        
        if isinstance(self.__criterion, nn.MSELoss):
            one_hot_labels = one_hot_encode(
                labels, self.__config['data']['num_classes'].to(self.__device)
            )
            loss = self.__criterion(logits, one_hot_labels)
        else:
            loss = self.__criterion(logits, labels)
            
        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        return loss, correct

    def __train(self, live_stats):
        """
        Perform training over one epoch
        """
        
        self.__model.train()
        train_loss = 0.0
        train_correct = 0.0

        for inputs, labels in self.__train_loader:
            inputs, labels = inputs.to(self.__device), labels.to(self.__device)
            loss, correct = self.__compute_loss_accuracy(inputs, labels)
            train_loss += loss.detach().cpu().item()
            train_correct += correct

            # Back Prop
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            # live stats
            current_acc = correct * 100 / len(labels)
            live_stats['loss'].append(loss.detach().cpu().item())
            live_stats['acc'].append(current_acc)

            if len(live_stats['loss']) > 100:
                live_stats['loss'].pop(0)
                live_stats['acc'].pop(0)

            live_stats["describer"].\
                set_description(f"Train (loss={np.mean(live_stats['loss']):.3f}, acc={np.mean(live_stats['acc']):.3f})")

        # return train_loss and train acc
        # (trainloader length = batch #, trainloader.dataset = image #)
        return train_loss / len(self.__train_loader), \
                train_correct * 100 / len(self.__train_loader.dataset)


    def __test(self, validation=False):
        self.__model.eval()
        test_loss = 0.0
        test_correct = 0.0

        if validation == True:
            dataloader = self.__val_loader
        else:
            dataloader = self.__test_loader

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                loss, correct = self.__compute_loss_accuracy(inputs, labels)
                test_loss += loss.detach().cpu().item()
                test_correct += correct

        return test_loss / len(dataloader), \
                test_correct * 100 / len(dataloader.dataset)

    # Following are some interaction code design
    def test_model(self):
        return self.__test()
                
    def get_model(self):
        return self.__model

    def get_device(self):
        return self.__device

    def load_model(self, model):
        self.__model = model

    def get_stats(self):
        return self.__train_losses, \
            self.__train_accs, \
            self.__val_losses, \
            self.__val_accs
    
    def get_config(self):
        return self.__config