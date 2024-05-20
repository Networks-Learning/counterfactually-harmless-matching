import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

def load_synthetic_data(
    data_dir='./data',
    refugee_batch_size=100,
    refugee_batch_num=5000,
    location_num=10,
):
    data_detail = f'synthetic_{refugee_batch_size}X{refugee_batch_num}_{location_num}_locations'
    data_path = Path(data_dir) / data_detail

    refugee_df = pd.read_csv(data_path / 'refugees.csv')
    capacity_df = pd.read_csv(data_path / 'capacity.csv')
    location_probs = np.load(data_path / 'location_probs.npy')
    employments = np.load(data_path / 'employments.npy')
    
    return refugee_df, capacity_df, location_probs, employments

def load_classifier_scores(
    save_dir='./save_dir',
    refugee_batch_size=100,
    refugee_batch_num=5000,
    location_num=10,
    beta=0.6
):
    location_detail = f'bias_{beta}_classifier_{refugee_batch_size}X{refugee_batch_num}_{location_num}_locations'
    score_dir = Path(save_dir) / 'scores' / location_detail
    train_scores = pd.read_csv(score_dir / 'train_scores.csv')
    valid_scores = pd.read_csv(score_dir / 'valid_scores.csv')
    test_scores = pd.read_csv(score_dir / 'test_scores.csv')
    
    return train_scores, valid_scores, test_scores

def load_assignment(
    data_dir='./data',
    refugee_batch_size=100,
    refugee_batch_num=5000,
    location_num=10,
    policy='maximum',
):
    data_detail = f'synthetic_{refugee_batch_size}X{refugee_batch_num}_{location_num}_locations'
    data_path = Path(data_dir) / data_detail
    
    assignments = np.load(data_path / f'{policy}_assignments.npy')
    
    return assignments


class CfHarmDataset(Dataset):
    def __init__(
        self,
        original_problems,
        new_problems,
        is_train: bool,
        max_len: int=100,
    ):
        self.x = [p.weight for p in original_problems]
        self.c = [p.capacity for p in original_problems]
        self.y = [p.weight for p in new_problems]
        
        self.is_train = is_train
        self.max_len = max_len
        
    def __len__(
        self
    ):
        return len(self.y)
    
    def __getitem__(
        self,
        i=int
    ):
        assert(self.x[i].shape == self.y[i].shape)
        
        cur_len, feat_dim = self.x[i].shape
        pad_len = self.max_len - cur_len
        
        pad = np.zeros([pad_len, feat_dim])
        x = np.concatenate([self.x[i], pad], axis=0)
        y = np.concatenate([self.y[i], pad], axis=0)
        c = self.c[i]
        
        mask_false = np.zeros(cur_len)
        mask_true = np.ones(pad_len)
        mask = np.concatenate([mask_false, mask_true], axis=0)
        
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        c = torch.tensor(c).float()
        mask = torch.tensor(mask).bool()
        
        return x, y, c, mask
    
def create_dataloader(
    original_problems,
    new_problems,
    is_train,
    batch_size,
    num_workers,
):
    ds = CfHarmDataset(
        original_problems,
        new_problems,
        is_train
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        persistent_workers=False,
        pin_memory=False,
    )
    
    return dl