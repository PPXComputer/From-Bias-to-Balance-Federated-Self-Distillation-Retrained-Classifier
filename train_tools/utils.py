from .models import *

__all__ = ["create_models"]


MODELS = {
    "fedavg_mnist": fedavgnet.FedAvgNetMNIST,

    "fedavg_tiny": fedavgnet.FedAvgNetTiny,
    "vgg11": vgg.vgg11,
    "res10": resnet.resnet10,
    "res18": resnet.resnet18,
    # cinic10
    "fedavg_cifar": fedavgnet.FedAvgNetCIFAR,
    "double_fedavg_cifar": fedavgnet.double_fedavg_cifar,
    # cifar10
    'resnet8': resnet.resnet8,
    'double_resnet8': resnet.double_resnet8,
    "super_constrast_resnet8": super_contrast_model.super_constrast_resnet8,
    # cifar100
    'resnet18': resnet.resnet_pretrain18,
    'double_resnet18': resnet.double_resnet_pretrain18,
    # tinyimagenet
    'resnet18nonorm': resnet.resnet18nonorm,
    'double_resnet18nonorm': resnet.double_resnet18nonorm,

}

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "cinic10": 10,
    "tinyimagenet": 200,
}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]
    model = MODELS[model_name](num_classes=num_classes)

    return model
