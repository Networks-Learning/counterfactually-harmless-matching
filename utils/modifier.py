import logging
import time
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from .common import AverageMeter
from .problem import Problem, sol2match
from .linear_program import create_primal, create_dual, solve_lp

class TransformerModifier(nn.Module):
    def __init__(
        self,
        model_type,
        input_dim,
        hidden_dim=512,
        n_tf_layers=6,
        n_pj_layers=2,
        n_head=1,
        epsilon=1.0,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.model_type = model_type
        
        cp_module_list = [nn.Linear(input_dim, hidden_dim)] + [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] * (n_pj_layers - 1)
        self.cp = nn.Sequential(*cp_module_list)
        
        wp_module_list = [nn.Linear(input_dim, hidden_dim)] + [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] * (n_pj_layers - 1)
        self.wp = nn.Sequential(*wp_module_list)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_head, batch_first=True)
        self.tf_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_tf_layers)
        
        ep_module_list = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_pj_layers - 1) + [nn.Linear(hidden_dim, input_dim)]
        self.ep = nn.Sequential(*ep_module_list)
        
    def forward(self, x, c, pad_mask):
        bsz, n_ref, n_loc = x.shape

        out = self.wp(x)
        ce = self.cp(c).unsqueeze(dim=1).repeat(1, n_ref, 1)
        out = out + ce
        
        tf_out = self.tf_encoder(out, src_key_padding_mask=pad_mask)
        
        delta = self.ep(tf_out)
        out = x + delta

        if self.training:
            final = None
        else:
            final = torch.clip(x + delta, min=0)
        
        return out, final
    
def get_model(
    model_type,
    input_dim,
    hidden_dim,
    n_tf_layers,
    n_pj_layers,
    epsilon,
):
    if model_type == 'tf':
        model = TransformerModifier(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_tf_layers=n_tf_layers,
            n_pj_layers=n_pj_layers,
            epsilon=epsilon,
        )
    else:
        assert(0)
        
    return model

class L2MatrixLoss(nn.Module):
    def __init__(
        self,
        reduction='mean'
    ):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, x, out, y, mask):
        assert(out.shape == y.shape)
        
        diff_mat = torch.square(out - y)
        bsz, n_refugee, hidden_dim = diff_mat.shape
        
        diff_mat = diff_mat.flatten(start_dim=0, end_dim=1)
        diff_mask = mask.flatten(start_dim=0, end_dim=1)
        
        diff_mat[diff_mask] = 0.
        diff_mat = diff_mat.reshape(bsz, n_refugee, hidden_dim)
        
        if self.reduction == 'mean':
            mat = diff_mat.sum(dim=-1).sum(dim=-1)
            denom = (1 - mask.int()).sum(dim=-1).float()

            mat = torch.div(mat, denom)

            loss = torch.mean(mat)
        elif self.reduction == 'sum':
            mat = diff_mat.sum(dim=-1).sum(dim=-1)

            loss = torch.mean(mat)
        
        return loss
    
def get_loss_func(
    loss_type='l2',
    reduction='mean',
):
    if loss_type == 'l2':
        loss_func = L2MatrixLoss(reduction=reduction)
    else:
        assert(0)
        
    return loss_func

def train_modifier(
    dir_path,
    model,
    train_dl,
    valid_dl,
    loss_type='l2',
    learning_rate=1e-3,
    epochs=10,
):
    loss_func = get_loss_func(loss_type=loss_type, reduction='sum')
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    optimizer.zero_grad()
    
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    min_valid_loss = 1e+8
    best_model_path = dir_path / 'model_best.pt'

    train_times = AverageMeter()
    valid_times = AverageMeter()
    
    dir_path.mkdir(parents=True, exist_ok=True)
    
    train_log = logging.getLogger()
    train_log.setLevel(logging.INFO)
    train_file_handler = logging.FileHandler(dir_path / 'log', mode='w')
    train_file_handler.setFormatter(logging.Formatter('%(message)s'))
    train_log.addHandler(train_file_handler)

    train_log.info('Start Training\n')
    
    for epoch in range(epochs):
        train_log.info(f'Epoch {epoch}')
        
        train_start = time.time()
        train_loss = AverageMeter()
        
        pbar = tqdm(train_dl)
        
        model.train()
        for batch in train_dl:
            x, y, c, mask = batch
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()
            mask = mask.cuda()
            bsz, n_token, n_loc = x.shape
            
            out, _ = model(x, c, mask)
            loss = loss_func(x, out, y, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=bsz)
            pbar.set_description(f'Epoch {epoch} Train Loss:{train_loss.avg: .3f}')
            
            pbar.update(1)
        pbar.close()
        scheduler.step()
        
        train_end = time.time()
        train_time = train_end - train_start
        train_log.info(f'Training Time: {train_time:.3f} s')
        train_times.update(train_time)

        valid_start = time.time()
        valid_loss = AverageMeter()
        
        pbar = tqdm(valid_dl)
        
        model.eval()
        with torch.inference_mode():
            for batch in valid_dl:
                x, y, c, mask = batch
                x = x.cuda()
                y = y.cuda()
                c = c.cuda()
                mask = mask.cuda()
                bsz = len(y)
                
                out, _ = model(x, c, mask)
                loss = loss_func(x, out, y, mask)
            
                valid_loss.update(loss.item(), n=bsz)
                pbar.set_description(f'Epoch {epoch} Valid Loss:{valid_loss.avg: .3f}')
                    
                pbar.update(1)
            pbar.close()
        
        valid_end = time.time()
        valid_time = valid_end - valid_start
        train_log.info(f'Validation Time: {train_time:.3f} s')
        valid_times.update(valid_time)
        
        train_log.info(f'Train Loss: {train_loss.avg:.3f}')
        train_log.info(f'Valid Loss: {valid_loss.avg:.3f}')
        train_log.info('\n')
        
        if valid_loss.avg < min_valid_loss:
            min_valid_loss = valid_loss.avg
            torch.save(model.state_dict(), str(best_model_path))
            
    train_log.info(f'Average Train Time: {train_times.avg:.3f} s')
    train_log.info(f'Average Valid Time: {valid_times.avg:.3f} s')

    train_log.removeHandler(train_file_handler)
    train_file_handler.close()
    
    return model

def get_match(problem):
    primal = create_primal(problem)
    primal_sol, primal_val = solve_lp(primal)
    match = sol2match(problem, primal_sol)
    
    return match

def get_metrics(locations, employment, location_prob, common_m, target_m):
    def get_mean(number_list):
        if len(number_list) == 0:
            return 0.0
        else:
            return np.array(number_list).mean()
        
    target_er = []
    target_ee = []
    target_cf = []
    
    for i, edge in enumerate(target_m):
        loc_index = locations.index(edge[1])
        er = employment[i][loc_index]
        ee = location_prob[i][loc_index]
        if edge in common_m:
            cf = er
        else:
            cf = ee
        
        target_er.append(er)
        target_ee.append(ee)
        target_cf.append(cf)
        
    target_employment_ratio = get_mean(target_er)
    target_expected_employment = get_mean(target_ee)
    target_counterfactual_employment = get_mean(target_cf)
    
    return target_employment_ratio, target_expected_employment, target_counterfactual_employment

def eval_modifier(
    dir_path,
    model,
    problems,
    new_problems,
    employments,
    location_probs,
    default_match,
    true_match,
    max_len=128,
    file_name='model_eval'
):
    eval_log = logging.getLogger()
    eval_log.setLevel(logging.INFO)
    eval_file_handler = logging.FileHandler(dir_path / file_name, mode='w')
    eval_file_handler.setFormatter(logging.Formatter('%(message)s'))
    eval_log.addHandler(eval_file_handler)

    default_match_dict = defaultdict(list)
    true_match_dict = defaultdict(list)
    clf_match_dict = defaultdict(list)
    algo_match_dict = defaultdict(list)
    dl_match_dict = defaultdict(list)
    
    name_list = ['Default', 'True', 'Biased Classifier', 'Algorithm', 'DL Model']
    dict_list = [default_match_dict, true_match_dict, clf_match_dict, algo_match_dict, dl_match_dict]
    
    model.eval()
    with torch.inference_mode():
        eval_start = time.time()

        for problem, new_problem, employment, location_prob, default_m, true_m in zip(
            tqdm(problems, desc='Evaluation'), new_problems, employments, location_probs, default_match, true_match
        ):
            default_m = [
                (problem.refugees[i], problem.locations[m]) for i, m in enumerate(default_m)
            ]
            true_m = [
                (problem.refugees[i], problem.locations[m]) for i, m in enumerate(true_m)
            ]

            x = problem.weight
            y = new_problem.weight
            c = problem.capacity
            
            cur_len, feat_dim = x.shape
            pad_len = max_len - cur_len
            pad = np.zeros([pad_len, feat_dim])
            
            x = np.concatenate([x, pad], axis=0)
            y = np.concatenate([y, pad], axis=0)
            
            mask_false = np.zeros(cur_len)
            mask_true = np.ones(pad_len)
            mask = np.concatenate([mask_false, mask_true], axis=0)
            
            x = torch.tensor(x).cuda().float()
            mask = torch.tensor(mask).cuda().bool()
            y = torch.tensor(y).cuda().float()
            c = torch.tensor(c).cuda().float()
            
            out, final = model(x.unsqueeze(dim=0), c.unsqueeze(dim=0), mask.unsqueeze(dim=0))

            final = final.squeeze(dim=0)
            final = final[~mask]

            pred_problem = Problem(
                real_refugees=problem.real_refugees,
                dummies=problem.dummies,
                locations=problem.locations,
                capacity=problem.capacity,
                weight=final.detach().cpu().numpy(),
            )
            
            clf_m, algo_m, dl_m = [
                get_match(p)
                for p in [problem, new_problem, pred_problem]
            ]

            match_list = [default_m, true_m, clf_m, algo_m, dl_m]
            
            for name, target_m, target_dict in zip(name_list, match_list, dict_list):
                common_m = list(set(target_m) & set(default_m))
                
                employment_ratio, expected_employment, counterfactual_employment = get_metrics(
                    problem.locations, employment, location_prob, common_m, target_m
                )

                target_dict['Employment Ratio'].append(employment_ratio)
                target_dict['Expected Employment'].append(expected_employment)
                target_dict['Counterfactual Employment'].append(counterfactual_employment)

    eval_end = time.time()
    eval_time = eval_end - eval_start
    
    for metric in ['Employment Ratio', 'Expected Employment', 'Counterfactual Employment']:
        for target_dict, name in zip(dict_list, name_list):
            target_metric = np.array(target_dict[metric])
            target_mean = target_metric.mean()
            target_stderr = target_metric.std() / (len(target_metric)**0.5)
            eval_log.info(f'{metric} ({name}): {target_mean:.4f} ({target_stderr:.4f})')
            with open(dir_path / f'dict_{name}.json', 'w') as f:
                json.dump(target_dict, f)
                
        eval_log.info('')

    eval_log.info(f'Evaluation Time: {eval_time:.3f} s')

    eval_log.removeHandler(eval_file_handler)
    eval_file_handler.close()