import torch
import os
import sys

from torch import nn

import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ContrastClientTrainer"]


def reweight_ce(target_logits, target_logits_softmax, spurious_logits, label, num_classes, EPS=1e-6):
    p_vanilla = target_logits_softmax
    p_spurious = torch.sigmoid(spurious_logits)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    ce_loss = criterion(target_logits, label)

    flattened_labels = label.long().flatten()
    for target_val in range(num_classes):
        indices = (flattened_labels == target_val).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue
        p_vanilla_w_same_t_val = p_vanilla[indices, target_val]
        p_spurious_w_same_t_val = p_spurious[indices, target_val]

        positive_spurious_group_avg_p = (p_spurious_w_same_t_val * p_vanilla_w_same_t_val).sum() / (
                p_spurious_w_same_t_val.sum() + EPS)
        negative_spurious_group_avg_p = ((1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val).sum() / (
                (1 - p_spurious_w_same_t_val).sum() + EPS)

        if negative_spurious_group_avg_p < positive_spurious_group_avg_p:
            p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val

        weight = p_spurious_w_same_t_val + 1
        ce_loss[indices] *= weight

    ce_loss = ce_loss.mean()
    return ce_loss


class ContrastClientTrainer(BaseClientTrainer):
    def __init__(self, dataset_name, **kwargs):
        super(ContrastClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        self.dataset_name = dataset_name
        self.lam1 = self.algo_params['lam1']
        self.lam2 = self.algo_params['lam2']
        self.kld = nn.KLDivLoss().to(self.device)

    def train(self):
        """Local training"""

        # Keep global model's weights
        # self._keep_global()

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()
                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                logits_classifer, logits_local_classifer = self.model(data)
                # update the student and teacher
                l1 = self.criterion(logits_classifer, targets)
                softmax_teacher = F.softmax(logits_classifer, dim=1)
                softmax_student = F.softmax(logits_local_classifer, dim=1)
                kloss = self.kld(softmax_teacher + 1e-11, softmax_student.detach() + 1e-11)
                loss = self.lam1 * l1 + self.lam2 * reweight_ce(logits_local_classifer, softmax_student,
                                                                logits_classifer.detach(), targets,
                                                                self.num_classes) + kloss
                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size

    def run_classifier_one_epoch(self, lr, momentum, weight_decay):
        self.model.train()
        self.model.to(self.device)
        self.model.zero_grad()
        for name, param in self.model.named_parameters():
            if name in ['classifier_head.weight', 'classifier_head.bias']:
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
                _, logits_head = self.model(data)
                loss = self.criterion(logits_head, targets)  # 交叉熵
                loss.backward()
                optimizer.step()

        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def download_global(self, server_weights, lr, momentum, decay):
        """Load model & Optimizer"""

        for name_param in reversed(server_weights):
            if name_param == 'classifier_head.bias':
                server_weights[name_param].copy_(server_weights['classifier.bias'])
            if name_param == 'classifier_head.weight':
                server_weights[name_param].copy_(server_weights['classifier.weight'])
                break
        self.model.load_state_dict(server_weights)
        self.run_classifier_one_epoch(lr=lr, momentum=momentum, weight_decay=decay)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
