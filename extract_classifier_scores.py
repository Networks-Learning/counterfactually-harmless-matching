from utils import common as cm
from utils import dataset as ds
from utils import classifier as clf
from utils import problem as pb

from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == "__main__":
    args = cm.parse_arguments()
    
    refugee_df, capacity_df, location_probs, employments = ds.load_synthetic_data(
        data_dir=args.data_dir,
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
    )

    clf_lp = clf.get_biased_location_probs(
        location_probs,
        beta=args.beta,
        seed=args.seed
    )
    
    train_clf_lp, valid_clf_lp, test_clf_lp = cm.split_data(
        clf_lp,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
    
    ref_id = refugee_df['refugee_id'].values
    train_ref_id, valid_ref_id, test_ref_id = cm.split_data(
        ref_id,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
    
    train_scores = clf.lp2df(train_clf_lp, train_ref_id)
    valid_scores = clf.lp2df(valid_clf_lp, valid_ref_id)
    test_scores = clf.lp2df(test_clf_lp, test_ref_id)
    
    location_detail = f'bias_{args.beta}_classifier'
    location_detail += f'_{args.refugee_batch_size}X{args.refugee_batch_num}_{args.location_num}_locations'
    score_path = Path(args.save_dir) / 'scores' / location_detail
    score_path.mkdir(exist_ok=True, parents=True)
    
    train_scores.to_csv(score_path  / 'train_scores.csv', index=False)
    valid_scores.to_csv(score_path / 'valid_scores.csv', index=False)
    test_scores.to_csv(score_path / 'test_scores.csv', index=False)
    
    clf_assignments = pb.make_assignments(
        clf_lp,
        capacity_df,
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
        policy='maximum',
        seed=args.seed
    )
    
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
    
    _, _, test_clf_assignments = cm.split_data(
        clf_assignments.flatten(),
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )

    avg_test_clf_probs = np.array([test_location_probs[i][l] for i, l in enumerate(test_clf_assignments)]).mean()
    avg_test_clf_emp = np.array([test_employments[i][l] for i, l in enumerate(test_clf_assignments)]).mean()
    print(f'Optimal Policy with Classifier Score: {avg_test_clf_probs:.4f} probability, {avg_test_clf_emp:.4f} utility')
    
    with open(score_path / 'eval', 'w') as f:
        print(f'Optimal Policy with Classifier Score: {avg_test_clf_probs:.4f} probability, {avg_test_clf_emp:.4f} utility', file=f)