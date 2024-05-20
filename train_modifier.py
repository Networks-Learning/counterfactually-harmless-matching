from pathlib import Path
import torch

from utils import common as cm
from utils import problem as pb
from utils import dataset as ds
from utils import modifier as mf

if __name__ == "__main__":
    args = cm.parse_arguments()
    cm.reset_seed(args.seed)
    
    if args.gpu >=0:
        gpu = torch.device(f'cuda:{args.gpu}')
    else:
        gpu = torch.device('cpu')
    torch.cuda.set_device(gpu)
    
    if args.epsilon == -1:
        epsilons = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
    else:
        epsilons = [args.epsilon]
        
    if args.w == -1:
        w_list = [i/8 for i in range(9)]
    else:
        w_list = [args.w]
        
    _, _, location_probs, employments = ds.load_synthetic_data(
        data_dir=args.data_dir,
        refugee_batch_size=args.refugee_batch_size,
        refugee_batch_num=args.refugee_batch_num,
        location_num=args.location_num,
    )
    
    _, _, test_location_probs = cm.split_data(
        location_probs,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
    test_location_probs = test_location_probs.reshape(-1, args.refugee_batch_size, args.location_num)
    
    _, _, test_employments = cm.split_data(
        employments,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
    test_employments = test_employments.reshape(-1, args.refugee_batch_size, args.location_num)
        
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
            
            train_problems, train_success_edges, new_train_problems = pb.load_problems(problems_path / 'train')
            valid_problems, valid_success_edges, new_valid_problems = pb.load_problems(problems_path / 'valid')
            test_problems, test_success_edges, new_test_problems = pb.load_problems(problems_path / 'test')
            
            train_dl = ds.create_dataloader(
                train_problems,
                new_train_problems,
                is_train=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            valid_dl = ds.create_dataloader(
                valid_problems,
                new_valid_problems,
                is_train=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            test_dl = ds.create_dataloader(
                test_problems,
                new_test_problems,
                is_train=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            
            model = mf.get_model(
                model_type=args.modifier,
                input_dim=train_dl.dataset[0][0].shape[1],
                n_tf_layers=args.n_tf_layers,
                n_pj_layers=args.n_pj_layers,
                hidden_dim=args.hidden_dim,
                epsilon=args.epsilon
            )
            model = model.cuda()
            
            dir_detail = f'{args.modifier}_modifier_{args.seed}'
            dir_path = Path(args.result_dir) / dir_detail / problems_detail / target

            print(f'Model Save Path: {dir_path}')
            
            if not args.only_eval:
                model = mf.train_modifier(
                    dir_path,
                    model,
                    train_dl,
                    valid_dl,
                    loss_type=args.loss,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs
                )
                
            best_model_path = dir_path / 'model_best.pt'
            model.load_state_dict(torch.load(best_model_path, map_location=gpu), strict=True)
            
            default_match = ds.load_assignment(
                refugee_batch_size=args.refugee_batch_size,
                refugee_batch_num=args.refugee_batch_num,
                location_num=args.location_num,
                policy=policy
            )
            _, _, test_default_match = cm.split_data(
                default_match,
                train_ratio=args.train_ratio,
                test_ratio=args.test_ratio
            )
            
            true_match = ds.load_assignment(
                refugee_batch_size=args.refugee_batch_size,
                refugee_batch_num=args.refugee_batch_num,
                location_num=args.location_num,
                policy='maximum'
            )
            _, _, test_true_match = cm.split_data(
                default_match,
                train_ratio=args.train_ratio,
                test_ratio=args.test_ratio
            )
            
            mf.eval_modifier(
                dir_path,
                model,
                problems=test_problems,
                new_problems=new_test_problems,
                employments=test_employments,
                location_probs=test_location_probs,
                default_match=test_default_match,
                true_match=test_true_match,
            )