import copy
import functools
import math
import os
import pickle

import numpy as np
import torch.nn.functional as F
# from train_tools.preprocessing.cinic10.loader import get_dataloader_cinic10,
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader

from train_tools.preprocessing.cifar10.datasets import CIFAR10_truncated
from train_tools.preprocessing.cifar100.datasets import CIFAR100_truncated
from train_tools.preprocessing.cifar100.loader import get_dataloader_cifar100, _data_transforms_cifar100
from train_tools.preprocessing.cifar10.loader import get_dataloader_cifar10, _data_transforms_cifar10
from train_tools.preprocessing.cinic10.datasets import ImageFolderTruncated
from train_tools.preprocessing.tinyimagenet.datasets import TinyImageNet_Truncated


def cifar100_data_distributed(client_dict, class_list, name_factor, alpha):
    n_clients = 20
    root = "data/cifar100"
    batch_size = 64
    data_distributed = {
        "global": {'train': get_dataloader_cifar100(root, train=True, batch_size=batch_size),
                   'test': get_dataloader_cifar100(root, train=False, batch_size=500)},
        "local": [
            {'train': get_dataloader_cifar100(root, train=True, batch_size=batch_size, dataidxs=client_dict[i]),
             'datasize': sum(class_list[i]), 'test': None, 'dist': None}
            for i in range(n_clients)],
        "data_map": np.array([class_list[i] for i in range(n_clients)]),
        "num_classes": 100,
    }
    pick = f'data/cifar100/{name_factor}_alpha_{alpha}_data_distributed.pick'
    os.makedirs(os.path.dirname(pick), exist_ok=True)
    with open(pick, 'wb') as f:
        pickle.dump(data_distributed, f)
    print(pick)


def pad_and_squeeze(x):
    return F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").data.squeeze()


def get_dataloader_cinic10(root, train=True, batch_size=50, dataidxs=None):
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            functools.partial(pad_and_squeeze),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),
        ]
    )

    # Transformer for test set
    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std), ]
    )

    if train:
        root = os.path.join(root, "train")
        dataset = ImageFolderTruncated(root, dataidxs, transform=train_transform)
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )

    else:
        root = os.path.join(root, "test")
        dataset = ImageFolderTruncated(root, dataidxs, transform=valid_transform)
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataset, dataloader


def cinic10_data_distributed(client_dict, class_list, name_factor, alpha):
    n_clients = 20
   
    batch_size = 50

    data_distributed = {
        "global": {'train': get_dataloader_cinic10(root, train=True, batch_size=batch_size)[1],
                   'test': get_dataloader_cinic10(root, train=False, batch_size=500)[1]},
        "local": [
            {'train': get_dataloader_cinic10(root, train=True, batch_size=batch_size, dataidxs=client_dict[i])[1],
             'datasize': sum(class_list[i]), 'test': None, 'dist': None}
            for i in range(n_clients)
        ],
        "data_map": np.array([class_list[i] for i in range(n_clients)]),
        "num_classes": 10,
    }
    pick = f'data/cinic10/{name_factor}_alpha_{alpha}_data_distributed.pick'
    os.makedirs(os.path.dirname(pick), exist_ok=True)
    with open(pick, 'wb') as f:
        pickle.dump(data_distributed, f)
    print(pick)


def tinyimagenet_if_01_alpha_05_data_distributed(client_dict, class_list):
    n_clients = 20
    
    # batch_size = 64

    data_distributed = {
        "global": {'train': get_dataloader_tinyimagenet(root, train=True, batch_size=batch_size)[1],
                   'test': get_dataloader_tinyimagenet(root, train=False, batch_size=500)[1]},
        "local": [
            {'train': get_dataloader_tinyimagenet(root, train=True, batch_size=batch_size, dataidxs=client_dict[i])[1],
             'datasize': sum(class_list[i]), 'test': None, 'dist': None}
            for i in range(n_clients)
        ],
        "data_map": np.array([class_list[i] for i in range(n_clients)]),
        "num_classes": num_classes,
    }
    pick = 'data/tinyimagenet/if_01_alpha_05_data_distributed.pick'
    os.makedirs(os.path.dirname(pick), exist_ok=True)
    with open(pick, 'wb') as f:
        pickle.dump(data_distributed, f)
    print(pick)
   



def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor ** (_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    return img_num_per_cls


def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')
    print(img_num_list)

    list_clients_indices = []
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))
    return img_num_list, list_clients_indices


def clients_indices(list_label2indices: list, num_classes: int, num_clients:
int, non_iid_alpha: float, seed=None):
    indices2targets = []
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients)
    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
    list_client2indices = partition_balance(indices_dirichlet, num_clients)

    return list_client2indices


def partition_balance(idxs, num_split: int):
    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def build_non_iid_by_dirichlet(
        seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)

    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
            from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        # n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                                  :-1
                                  ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch


def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        print(f'{client}: {nums_data}')
    return dict_per_client


def tinyimagenet_transforms(x):
    return F.pad(x.unsqueeze(0), (8, 8, 8, 8), mode="reflect").data.squeeze()


def get_dataloader_tinyimagenet(root, train=True, batch_size=50, dataidxs=None):
    tinyimagenet_mean = [0.485, 0.456, 0.406]
    tinyimagenet_std = [0.229, 0.224, 0.225]
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            functools.partial(tinyimagenet_transforms),
            transforms.ToPILImage(),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std),
        ]
    )

    # Transformer for test set
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std),
        ]
    )
    if train:
        dataset = TinyImageNet_Truncated(
            root, True, dataidxs, transform=train_transform
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )
    else:
        dataset = TinyImageNet_Truncated(
            root, False, dataidxs, transform=valid_transform
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )
    return dataset, dataloader


def cifar10_data_distributed(client_dict, class_list, name_factor, alpha):
    n_clients = 20
    
    batch_size = 32

    data_distributed = {
        "global": {'train': get_dataloader_cifar10(root, train=True, batch_size=batch_size),
                   'test': get_dataloader_cifar10(root, train=False, batch_size=500)},
        "local": [
            {'train': get_dataloader_cifar10(root, train=True, batch_size=batch_size, dataidxs=client_dict[i]),
             'datasize': sum(class_list[i]), 'test': None, 'dist': None}
            for i in range(n_clients)
        ],
        "data_map": np.array([class_list[i] for i in range(n_clients)]),
        "num_classes": 10,
    }
    pick = f'data/cifar10/{name_factor}_alpha_{alpha}_data_distributed.pick'
    os.makedirs(os.path.dirname(pick), exist_ok=True)
    with open(pick, 'wb') as f:
        pickle.dump(data_distributed, f)
    print(pick)


def get_dataloader_cifar10_local(root, train=True, batch_size=50, dataidxs=None):
    train_transform, valid_transform = _data_transforms_cifar10()

    if train:
        dataset = CIFAR10_truncated(
            root, dataidxs, train=True, transform=train_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )


    else:
        dataset = CIFAR10_truncated(
            root, dataidxs, train=False, transform=valid_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataset, dataloader


def get_dataloader_cifar100_local(root, train=True, batch_size=50, dataidxs=None):
    train_transform, valid_transform = _data_transforms_cifar100()
    if train:
        dataset = CIFAR100_truncated(
            root, dataidxs, train=True, transform=train_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )

    else:
        dataset = CIFAR100_truncated(
            root, dataidxs, train=False, transform=valid_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataset, dataloader


if __name__ == "__main__":

    # StepLR()
    name_list = ['cifar10', 'cinic10', 'cifar100']
    root_list = ["data/cifar/",
                 "data/cinic10/",
                 "data/cifar100/"]
    for root, name in zip(root_list, name_list):
        for imb_factor, name_factor in zip([0.01, 0.02, 0.1],
                                           ['if_001', 'if_002', 'if_01']):
            # imb_factor = 0.1
            non_iid_alpha = 10
            if name == 'cifar10':
                batch_size = 32
                num_classes = 10
                dataset, dataloader = get_dataloader_cifar10_local(root, True, batch_size=batch_size)
            elif name == 'cifar100':
                batch_size = 64
                num_classes = 100
                dataset, dataloader = get_dataloader_cifar100_local(root, True, batch_size=batch_size)
            else:
                batch_size = 50
                num_classes = 10
                dataset, dataloader = get_dataloader_cinic10(root, True, batch_size=batch_size)
            num_clients = 20
            list_label2indices = classify_label(dataset, num_classes)
            _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), num_classes,
                                                              imb_factor, 'exp')
            list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), num_classes,
                                                  num_clients, non_iid_alpha, 7)
            all_list_indcies = []
            for client_indices in list_client2indices:
                for i in client_indices:
                    all_list_indcies.append(i)
            print(len(all_list_indcies) == len(set(all_list_indcies)))
            original_dict_per_client = show_clients_data_distribution(dataset, list_client2indices,
                                                                      num_classes)
            if name == 'cifar10':
                cifar10_data_distributed(list_client2indices, original_dict_per_client, name_factor,
                                         non_iid_alpha)
            elif name == 'cifar100':
                cifar100_data_distributed(list_client2indices, original_dict_per_client, name_factor,
                                          non_iid_alpha)
            else:
                cinic10_data_distributed(list_client2indices, original_dict_per_client, name_factor,
                                         non_iid_alpha)
