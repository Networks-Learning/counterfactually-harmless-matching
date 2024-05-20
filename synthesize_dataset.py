from utils import synthesizer as sz
from utils import common as cm
from utils import problem as pb

from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == "__main__":
    args = cm.parse_arguments()
    
    if args.w == -1:
        w_list = [i/8 for i in range(9)]
    else:
        w_list = [args.w]
    
    refugee_df, loc_df = sz.synthesize_refugee(
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
        seed=args.seed
    )
    
    capacity_df = sz.synthesize_capacity(
        loc_df,
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
        seed=args.seed
    )
    
    location_probs, mu_aceg_l, empirical_mu_aceg_l = sz.synthesize_location_probs(
        refugee_df=refugee_df,
        location_num=args.location_num,
        seed=args.seed
    )
    
    employments = sz.synthesize_employment(
        location_probs,
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
        seed=args.seed
    )
    
    train_df, valid_df, test_df = cm.split_data(
        refugee_df,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
    
    data_dir = Path(args.data_dir)
    data_detail = f'synthetic_{args.refugee_batch_size}X{args.refugee_batch_num}_{args.location_num}_locations'
    data_path = data_dir / data_detail
    data_path.mkdir(exist_ok=True, parents=True)
    
    refugee_df.to_csv(data_path / 'refugees.csv', index=False)
    train_df.to_csv(data_path / 'train.csv', index=False)
    valid_df.to_csv(data_path / 'valid.csv', index=False)
    test_df.to_csv(data_path / 'test.csv', index=False)
    
    capacity_df.to_csv(data_path / 'capacity.csv', index=False)
    
    np.save(data_path / 'location_probs.npy', location_probs)
    np.save(data_path / 'mu_aceg_l.npy', mu_aceg_l)
    np.save(data_path / 'empirical_mu_aceg_l.npy', empirical_mu_aceg_l)
    
    np.save(data_path / 'employments.npy', employments)
        
    maximum_assignment = pb.make_assignments(
        location_probs,
        capacity_df,
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
        policy='maximum',
        seed=args.seed
    )
    np.save(data_path / 'maximum_assignments.npy', employments)
        
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
                
    result_df = []
    for w in w_list:
        policy = f'maximum_{w}'
        print(f'Making assignment of policy {policy} (Noise Level w={w})')
        
        assignments = sz.shuffle_assignment(
            maximum_assignment,
            noise_ratio=w,
            seed=0
        )
        
        np.save(data_path / f'{policy}_assignments.npy', assignments)
            
        _, _, test_assignments = cm.split_data(
            assignments.flatten(),
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio
        )
        
        avg_probs = np.array([test_location_probs[i][l] for i, l in enumerate(test_assignments)]).mean()
        avg_emp = np.array([test_employments[i][l] for i, l in enumerate(test_assignments)]).mean()
            
        result_df.append([policy, avg_probs, avg_emp])
        
    result_df = pd.DataFrame(result_df, columns=['policy', 'avg prob', 'utility'])
    result_df.to_csv(data_path / 'default_policy_evaluation.csv', index=False)