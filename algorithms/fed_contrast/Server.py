import copy
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.fed_contrast.ContrastClientTrainer import ContrastClientTrainer

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
            self, algo_params, model, data_distributed, optimizer, scheduler=None, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        self.dataset_name = self.algo_params['dataset_name']
        self.client = ContrastClientTrainer(
            self.dataset_name,
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )
        logging.info("\n>>> Fed-focal Loss Server initialized...\n")

    def _clients_training(self, sampled_clients):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        # server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            momentum, decay = self.get_momentum_decay()
            cur_epoch_lr = self.scheduler.get_last_lr()[-1]
            print(f'cur:{cur_epoch_lr}')
            self.client.download_global(server_weights, cur_epoch_lr, momentum, decay)

            # Local training
            local_results, local_size = self.client.train()

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results

    def get_momentum_decay(self):
        if self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
            return 0, 0
        elif self.dataset_name == 'cinic10':
            return 0.9, 1e-5
        elif self.dataset_name == 'tinyimagenet':
            return 0.9, 0.001
        else:
            raise NotImplementedError(f'dataset:{self.dataset_name}')

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.n_rounds):

            # Initial Model Statistics
            if round_idx == 0:
                test_acc = evaluate_model(
                    self.model, self.testloader, device=self.device, numclass=self.num_classes
                )
                self.server_results["test_accuracy"].append(test_acc)

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling()
            self.server_results["client_history"].append(sampled_clients)

            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients
            )

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Update global weights and evaluate statistics
            self._update_and_evaluate(ag_weights, round_results, round_idx, start_time)

    def _update_and_evaluate(self, ag_weights, round_results, round_idx, start_time):
        """Evaluate experiment statistics."""

        # Update Global Server Model
        self.model.load_state_dict(ag_weights)

        # Measure Accuracy Statistics
        test_acc = evaluate_model(self.model, self.testloader, device=self.device, numclass=self.num_classes)
        self.server_results["test_accuracy"].append(test_acc)

        # # Evaluate Personalized FL performance 去掉本地化
        # eval_results = get_round_personalized_acc(
        #     round_results, self.server_results, self.data_distributed
        # )
        # wandb.log(eval_results, step=round_idx)

        # Change learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        round_elapse = time.time() - start_time

        # Log and logging.info

        self._print_stats(round_results, test_acc, round_idx, round_elapse)
        logging.info("-" * 50)


@torch.no_grad()
def evaluate_model(model, dataloader, device="cuda:0", numclass=10):
    """Evaluate model accuracy for the given dataloader"""
    model.eval()
    model.to(device)

    running_count = 0
    running_correct = 0

    # For per-class accuracy, initialize counters
    num_classes = numclass  # Assuming dataset has 'classes' attribute
    # class_correct = [0] * num_classes
    # class_total = [0] * num_classes

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        logits,_ = model(data)
        pred = logits.max(dim=1)[1]

        running_correct += (targets == pred).sum().item()
        running_count += data.size(0)

        # Update per-class counters
        # for t, p in zip(targets, pred):
        #     class_correct[t] += (t == p).item()
        #     class_total[t] += 1

    accuracy = round(running_correct / running_count, 4)

    # Compute per-class accuracies
    # class_accuracies = [round(class_correct[i] / class_total[i], 4) if class_total[i] != 0 else 0 for i in
    #                     range(num_classes)]
    #
    # # Print per-class accuracies
    # logging.info(f"Accuracy for class {class_accuracies}%")
    return accuracy
