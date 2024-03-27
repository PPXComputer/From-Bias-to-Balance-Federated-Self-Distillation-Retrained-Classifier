# From Bias to Balance: Federated Self-Distillation Retrained Classifier


This project is written in Python for generating dataset distributions and launching federated learning algorithms.

## Dataset Generation 

First, modify the `root_list` variable in the `load_pick.py` file to point to your own dataset paths. For example:

```python
root_list = ["data/cifar10/", 
             "data/cifar100/",
             "data/cinic10/"]
```

Then run the `load_pick.py` file to generate the dataset distribution.

## Algorithm Startup

Use the following commands to launch the federated learning algorithm:

### FedSDC Algorithm
```
python main.py --config_path /path/to/config/fed_classifier_if01.json --partition_method [if_01|if_001|if_002]
```

### FedSDC-$\omega$ Algorithm
```
python main.py --config_path /path/to/config/fed_more_contrast_if01.json --partition_method [if_01|if_001|if_002]
```

Where:

- `--config_path` specifies the configuration file path
- `--partition_method` specifies the data partitioning method
  - `if_01` indicates an imbalance factor of 10
  - `if_001` indicates an imbalance factor of 100
  - `if_002` indicates a balance factor of 50

With different `partition_method` parameters, you can obtain different degrees of data imbalance distribution to test the robustness of the algorithm.

Make sure to replace the paths in the commands with your own configuration file paths.

This README file briefly introduces how to generate dataset distributions and launch the FedSDC and FedSDC-$\omega$ federated learning algorithms. If you have any other questions, feel free to inquire.