import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

    def run_classifier_one_epoch(self, lr, momentum, weight_decay):
        self.model.train()
        self.model.to(self.device)
        self.model.zero_grad()
        for name, param in self.model.named_parameters():
            if name in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = True
            else:
                param.requires_grad = False

        epoch_classifier = self.algo_params['epoch_classifier']

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                    momentum=momentum, weight_decay=weight_decay)

        for ep in range(epoch_classifier):
            # local training for 1 epoch
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.model.zero_grad()
                logits_head = self.model(data)
                loss = self.criterion(logits_head, targets)  # 交叉熵
                loss.backward()
                optimizer.step()

        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def download_global(self, server_weights, lr, momentum, decay):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.run_classifier_one_epoch(lr=lr, momentum=momentum, weight_decay=decay)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)