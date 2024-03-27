import copy
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.fed_classifier.ClientTrainer import ClientTrainer
from algorithms.BaseServer import BaseServer

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
            self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )
        self.dataset_name = self.algo_params['dataset_name']
        logging.info("\n>>> Fed classifier Server initialized...\n")

    def get_momentum_decay(self):
        if self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
            return 0, 0
        elif self.dataset_name == 'cinic10':
            return 0.9, 1e-5
        elif self.dataset_name == 'tinyimagenet':
            return 0.9, 0.001
        else:raise NotImplementedError(f'dataset:{self.dataset_name}')

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
