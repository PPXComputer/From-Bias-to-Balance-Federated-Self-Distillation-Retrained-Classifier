import logging

import torch
import torch.nn as nn
import numpy as np
import copy
import time

from .measures import *

__all__ = ["BaseServer"]


class BaseServer:
    def __init__(
            self,
            algo_params,
            model,
            data_distributed,
            optimizer,
            scheduler,
            n_rounds=200,
            sample_ratio=0.1,
            local_epochs=5,
            device="cuda:0",
    ):
        """
        Server class controls the overall experiment.
        """
        self.algo_params = algo_params
        self.num_classes = data_distributed["num_classes"]
        self.model = model
        self.testloader = data_distributed["global"]["test"]
        self.criterion = nn.CrossEntropyLoss()
        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_rounds = n_rounds
        self.device = device
        if isinstance(data_distributed['local'], list):
            self.n_clients = len(data_distributed["local"])
        else:
            self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs
        self.server_results = {
            "client_history": [],
            "test_accuracy": [],
        }
        self.random_state = np.random.RandomState(7)

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

    def _clients_training(self, sampled_clients):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(server_weights, server_optimizer)

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

    def _client_sampling(self):
        """Sample clients by given sampling ratio"""

        # make sure for same client sampling for fair comparison
        # np.random.seed(round_idx)
        clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
        sampled_clients = self.random_state.choice(
            self.n_clients, clients_per_round, replace=False
        )

        return sampled_clients

    def _set_client_data(self, client_idx):
        """Assign local client datasets."""
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.testloader = self.data_distributed["global"]["test"]

    def _aggregation(self, w, ns):
        """Average locally trained model parameters"""
        prop = torch.tensor(ns, dtype=torch.float)
        prop /= torch.sum(prop)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * prop[0]

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * prop[i]

        return copy.deepcopy(w_avg)

    def _results_updater(self, round_results, local_results):
        """Combine local results as clean format"""

        for key, item in local_results.items():
            if key not in round_results.keys():
                round_results[key] = [item]
            else:
                round_results[key].append(item)

        return round_results

    def _print_start(self):
        """logging.info initial log for experiment"""

        if self.device == "cpu":
            return "cpu"
        device_idx = None
        if isinstance(self.device, str):
            if self.device[-1] == '0' or self.device[-1] == '1':
                device_idx = int(self.device[-1])
        elif isinstance(self.device, torch._device):
            device_idx = self.device.index

        device_name = torch.cuda.get_device_name(device_idx)
        logging.info("")
        logging.info("=" * 50)
        logging.info("Train start on device: {}".format(device_name))
        logging.info("=" * 50)

    def _print_stats(self, round_results, test_accs, round_idx, round_elapse):
        logging.info(
            "[Round {}/{}] Elapsed {}s (Current Time: {})".format(
                round_idx + 1,
                self.n_rounds,
                round(round_elapse, 1),
                time.strftime("%H:%M:%S"),
            )
        )
        # logging.info(
        #     "[Local Stat (Train Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
        #         round_results["train_acc"],
        #         np.mean(round_results["train_acc"]),
        #         np.std(round_results["train_acc"]),
        #     )
        # )
        #
        # logging.info(
        #     "[Local Stat (Test Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
        #         round_results["test_acc"],
        #         np.mean(round_results["test_acc"]),
        #         np.std(round_results["test_acc"]),
        #     )
        # )

        logging.info(f"round_idx - {round_idx} [Server Stat] Acc - {test_accs}")

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
