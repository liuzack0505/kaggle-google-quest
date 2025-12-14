import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import TARGETS, N_TARGETS
from custom_datasets import TextDataset
from tokenization import tokenize
from inference import infer 
from train import models, pretrained_models
from create_features import get_ohe_categorical_features

def build_parser():
    parser = argparse.ArgumentParser(description='Inference on Test Set')
    parser.add_argument('-model_name', type=str, default='siamese_bert')
    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('-data_dir', type=str, default='data/')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-enable_loss_fn_weights', action='store_true')
    parser.add_argument('-enable_awp', action='store_true') 
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    
    model_name = args.model_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    suffix_mapping = {
        'enable_loss_fn_weights': 'weight',
        'enable_awp': 'awp'
    }
    folder_name_parts = [model_name]
    for arg_name, suffix in suffix_mapping.items():
        if getattr(args, arg_name, False):
            folder_name_parts.append(suffix)
    folder_name = '_'.join(folder_name_parts)
    
    # checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name) + '/'
    checkpoint_dir = args.checkpoint_dir
    print(f"Loading checkpoints from: {checkpoint_dir}")

    # Load data
    print("Loading Data...")
    train = pd.read_csv(f'{args.data_dir}train.csv')
    test = pd.read_csv(f'{args.data_dir}test.csv')
    sample_submission = pd.read_csv(f'{args.data_dir}sample_submission.csv')

    # Tokenization
    print("Tokenizing Test Data...")
    ids_test, seg_ids_test = tokenize(test, pretrained_model_str=pretrained_models[model_name])

    # Categorical features
    print("Processing Categorical Features...")
    _, cat_features_test = get_ohe_categorical_features(train, test, 'category')

    # Create dataloader
    dummy_y = np.zeros((len(test), N_TARGETS))
    
    test_loader = DataLoader(
        TextDataset(cat_features_test, ids_test['question'], ids_test['answer'],
                    seg_ids_test['question'], seg_ids_test['answer'], 
                    np.arange(len(test)), dummy_y),
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # K-Fold inference
    n_folds = 10
    final_preds = np.zeros((len(test), N_TARGETS))
    
    print(f"Starting Inference with {n_folds} folds...")

    for fold_id in range(n_folds):
        print(f'Predicting Fold {fold_id + 1}/{n_folds} ...')
        
        # Initialize model
        model = models[model_name]()
        model.to(device)
        
        # Load checkpoint
        ckpt_path = f'{checkpoint_dir}{model_name}_tuned_fold_{fold_id+1}_best.pth'
        
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found {ckpt_path}, skipping fold.")
            continue

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.float() 
        model.eval()

        fold_preds = infer(model, test_loader, checkpoint_file=None, device=device)
        
        final_preds += fold_preds

    final_preds /= n_folds
    
    submission = sample_submission.copy()
    submission[TARGETS] = final_preds
    
    output_filename = 'submission.csv'
    submission.to_csv(output_filename, index=False)
    print(f"Inference complete! Submission saved to {output_filename}")