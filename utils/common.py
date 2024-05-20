import random
import numpy as np
import torch
import pandas as pd
import argparse
import copy
from tqdm import tqdm

LOCATIONS = [
    'California', 'Florida', 'Illinois', 'Maryland', 'Massachusetts',
    'New Jersey', 'New York', 'Pennsylvania', 'Texas', 'Virginia'
]

def parse_arguments(
    return_default: bool=False,
) -> object:
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=0,
         help='Random seed'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU index to use in training of weight modifier'
    )
    parser.add_argument(
        '--refugee_batch_size', type=int, default=100,
        help='Refugee batch size'
    )
    parser.add_argument(
        '--refugee_batch_num', type=int, default=5000,
        help='Number of refugee batches'
    )
    parser.add_argument(
        '--location_num', type=int, default=10,
        help='Number of locations'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Ratio of train dataset'
    )
    parser.add_argument(
        '--test_ratio', type=float, default=0.1,
        help='Ratio of test dataset'
    )
    parser.add_argument(
        '--beta', type=float, default=0.6,
        help='Bias of classifier'
    )
    parser.add_argument(
        '--epsilon', type=float, default=-1,
        help='Epsilon in post-processing algorithm'
    )
    parser.add_argument(
        '--w', type=float, default=-1,
        help='Noise level in default policy'
    )
    parser.add_argument(
        '--modifier', type=str, default='tf',
        help='The type of weight modifier to train'
    )
    parser.add_argument(
        '--loss', type=str, default='l2',
        help='Loss function used in training'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=128,
        help='The hidden dimension'
    )
    parser.add_argument(
        '--n_tf_layers', type=int, default=2,
        help='The number of layers in transformer'
    )
    parser.add_argument(
        '--n_pj_layers', type=int, default=2,
        help='The number of layers in projection'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of workers for preprocessing'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of epochs'
    )
    parser.add_argument(
        '--only_eval', action='store_true',
        help='Do only evaluation'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data',
        help='Directory to load and save the synthetic data'
    )
    parser.add_argument(
        '--save_dir', type=str, default='./save_dir',
        help='Directory to save intermediate outputs'
    )
    parser.add_argument(
        '--result_dir', type=str, default='./result',
        help='Directory to save training results'
    )

    if return_default:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    return args

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(
        self
    ):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self,
        val: float,
        n: int=1
    ):
        """
        Update the values
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reset_seed(
    seed: int=7
):
    """
    Reset the random variables with the given seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def split_data(
    data,
    train_ratio=0.8,
    test_ratio=0.1
):
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_valid = n_total - n_train - n_test
    
    if type(data) == pd.DataFrame:
        train_data = data.iloc[:n_train]
        valid_data = data.iloc[n_train:n_train+n_valid].reset_index(drop=True)
        test_data = data.iloc[n_train+n_valid:].reset_index(drop=True)
    else:
        train_data = data[:n_train]
        valid_data = data[n_train:n_train+n_valid]
        test_data = data[n_train+n_valid:]
        
    return train_data, valid_data, test_data

    
def calculate_beta_distribution_parameter(
    mu,
    var,
):
    v = (mu * (1 - mu) / var) - 1
    concentration_alpha = mu * v
    concentration_beta = (1 - mu) * v
            
    return concentration_alpha, concentration_beta