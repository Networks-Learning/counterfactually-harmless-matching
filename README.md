# Matchings, Predictions and Counterfactual Harm in Refugee Resettlement Processes

This repo is the PyTorch code of our paper, which is currently under review.

## Required packages ##
- python == 3.9.13
- numpy == 1.24.3
- pandas == 1.5.3
- scipy == 1.10.1
- torch == 1.10.0
- tqdm == 4.65.0
  
## To run the code ##
Run synthesize_dataset.py for generation of synthetic dataset of refugees.
The generation results will be saved in a directory 'data/synthetic_100X5000_10_locations'.
```
python synthesize_dataset.py --refugee_batch_size [number of refugee in a batch] --refugee_batch_num [number of refugee batches]
```

Execute extract_classifier_scores.py to extract employment probability of each refugee predicted by biased classifier.
The extracted employment probability will be saved in a directory 'save_dir/scores'
```
python extract_classifier_scores.py --beta [bias of the classifier]
```

Then, start our proposed postprocessing framework by computing minimally modified weight that is guaranteed to be counterfactually harmless.
The adjusted weight will be saved in a directory 'save_dir/problems'
```
python create_problems.py --epsilon [epsilon used in the algorithm] --w [noise level of the default policy]
```

Train the deep learning model that learns to avoid counterfactual harm.
The trained model and evaluation result will be saved in a directory 'result'
```
python train_modifier.py
```
