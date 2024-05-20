import numpy as np
from pathlib import Path
import pickle
import torch
from collections import defaultdict
import pandas as pd

from utils import common as cm
from utils import problem as pb
from utils import dataset as ds

if __name__ == "__main__":
    args = cm.parse_arguments()
    cm.reset_seed(args.seed)
    
    if args.epsilon == -1:
        epsilons = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
    else:
        epsilons = [args.epsilon]
        
    if args.w == -1:
        w_list = [i/8 for i in range(9)]
    else:
        w_list = [args.w]
    
    train_scores, valid_scores, test_scores = ds.load_classifier_scores(
        save_dir=args.save_dir,
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
        beta=args.beta,
    )
    
    refugee_df, capacity_df, location_probs, employments = ds.load_synthetic_data(
        data_dir=args.data_dir,
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
    )
    ref_id = refugee_df['refugee_id'].values
    
    _, _, test_location_probs = cm.split_data(
        location_probs,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
    
    _, _, test_employments = cm.split_data(
        employments,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
    
    _, _, test_capacity_df = cm.split_data(
        capacity_df,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )

    train_problems = pb.get_batch_problem(train_scores, capacity_df)
    valid_problems = pb.get_batch_problem(valid_scores, capacity_df)
    test_problems = pb.get_batch_problem(test_scores, capacity_df)
    
    problems_dir = Path(f'{args.save_dir}') / 'problems'
    problems_detail = f'noise_{args.beta}_classifier'
    problems_detail += f'_{args.refugee_batch_size}X{args.refugee_batch_num}_{args.location_num}_locations'
    print(f'Problems Path: {problems_dir / problems_detail}')
    
    for epsilon in epsilons:
        for w in w_list:
            policy = f'maximum_{w}'
            
            target = f'{policy}_policy_{epsilon}_epsilon'
            problems_path = problems_dir / problems_detail / target
            print(target)

            assignment = ds.load_assignment(
                refugee_batch_size=args.refugee_batch_size,
                refugee_batch_num=args.refugee_batch_num,
                location_num=args.location_num,
                policy=policy
            )

            success_edges = pb.get_success_edges(
                ref_id,
                assignment,
                employments
            )
            
            train_success_edges, valid_success_edges, test_success_edges = cm.split_data(
                success_edges,
                train_ratio=args.train_ratio,
                test_ratio=args.test_ratio
            )

            pb.get_adjusted_weights(
                problem_save_path=problems_path / 'train',
                problems=train_problems,
                success_edges=train_success_edges,
                epsilon=epsilon,
            )

            pb.get_adjusted_weights(
                problem_save_path=problems_path / 'valid',
                problems=valid_problems,
                success_edges=valid_success_edges,
                epsilon=epsilon,
            )
            
            pb.get_adjusted_weights(
                problem_save_path=problems_path / 'test',
                problems=test_problems,
                success_edges=test_success_edges,
                epsilon=epsilon,
            )