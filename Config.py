import torch
import torch.nn as nn

from Model import *

class Config:
    def __init__(self, 
                 batch_size=128,
                 lr=0.001, 
                 num_comm=40, 
                 num_classes=10, 
                 num_clients=5,
                 num_local_steps=1, 
                 fed_algorithm='FedAvg',
                 mu=0.01,
                 criterion=nn.CrossEntropyLoss(),
                 optimizer='adam',
                 distribution='non-IID',
                 alpha=1,
                 model_type=ResNet50) -> None:
        self.batch_size = batch_size
        self.learning_rate = lr
        self.num_communications = num_comm
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.num_local_steps = num_local_steps
        self.fed_algorithm = fed_algorithm  # Options: 'FedAvg', 'FedProx', 'SCAFFOLD'
        self.mu = mu                        # for FedProx, usually 0.001 - 0.1
        self.criterion = criterion
        self.optimizer = optimizer
        self.distribution = distribution    # Options: 'IID', 'non-IID'
        self.alpha = alpha                  # control the level of non-IIDness
        self.model_type = model_type
