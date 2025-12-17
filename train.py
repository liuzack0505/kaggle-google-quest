from ast import parse
import os
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW


from common import TARGETS, N_TARGETS
from utilities.helpers import init_logger, init_seed
from custom_datasets import TextDataset
from tokenization import tokenize
from learning import Learner
from one_cycle import OneCycleLR
from create_features import get_ohe_categorical_features
from evaluation import spearmanr_torch, get_cvs
from inference import infer
from awp import AWP

from models.siamese_transformers import SiameseBert, SiameseRoberta, SiameseXLNet
from models.double_transformers import DoubleAlbert


models = {
    'siamese_bert': SiameseBert,
    'siamese_roberta': SiameseRoberta,
    'siamese_xlnet': SiameseXLNet,
    'double_albert': DoubleAlbert
}


pretrained_models = {
    # 'siamese_bert': 'bert-base-uncased',
    'siamese_roberta': 'roberta-base',
    'siamese_xlnet': 'xlnet-base-cased',
    'double_albert': 'albert-base-v2',

    'siamese_bert': './bert-base-uncased',
}


def get_optimizer_param_groups(model, lr, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': lr}
    ]
    return optimizer_grouped_parameters


def get_optimizer(model, lr, weight_decay, model_type='siamese'):
    param_groups = get_optimizer_param_groups(model.head, lr, weight_decay)
    if model_type == 'siamese':
        param_groups += get_optimizer_param_groups(
            model.transformer, lr / 100, weight_decay)
    elif model_type == 'double':
        param_groups += get_optimizer_param_groups(
            model.q_transformer, lr / 100, weight_decay)
        param_groups += get_optimizer_param_groups(
            model.a_transformer, lr / 100, weight_decay)
    return AdamW(param_groups)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Perform first stage of training.')
    parser.add_argument('-model_name', type=str, default='siamese_roberta')
    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('-log_dir', type=str, default='logs/')
    parser.add_argument('-data_dir', type=str, default='data/')
    parser.add_argument('-enable_loss_fn_weights', action='store_true')
    parser.add_argument('-enable_awp', action='store_true',
                        help='Enable Adversarial Weight Perturbation')
    parser.add_argument('-use_attention_pooling', action='store_true',
                        help='Use attention pooling instead of average pooling')
    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    model_name = args.model_name
    model_type = 'double' if model_name == 'double_albert' else 'siamese'

    # Define mapping from arguments to folder name suffixes
    suffix_mapping = {
        'enable_loss_fn_weights': 'weight',
        'enable_awp': 'awp',
        'use_attention_pooling': 'attnpool'
    }

    # Build folder name based on model and enabled features
    folder_name_parts = [model_name]
    for arg_name, suffix in suffix_mapping.items():
        if getattr(args, arg_name, False):
            folder_name_parts.append(suffix)
    folder_name = '_'.join(folder_name_parts)

    # Set checkpoint, log, and OOF directories
    checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name) + '/'
    log_dir = os.path.join(args.log_dir, folder_name) + '/'
    oof_dir = os.path.join('oofs/', folder_name) + '/'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    main_logger = init_logger(log_dir, f'train_main_{model_name}.log')

    # Import data
    test = pd.read_csv(f'{args.data_dir}test.csv')
    train = pd.read_csv(f'{args.data_dir}train.csv')

    # Min Max scale target after rank transformation
    for col in TARGETS:
        train[col] = train[col].rank(method="average")
    train[TARGETS] = MinMaxScaler().fit_transform(train[TARGETS])
    y = train[TARGETS].values

    # Get model inputs
    ids_train, seg_ids_train = tokenize(
        train, pretrained_model_str=pretrained_models[model_name])
    cat_features_train, _ = get_ohe_categorical_features(
        train, test, 'category')

    # Set training parameters
    device = 'cuda'
    num_workers = 10
    n_folds = 10
    lr = 0.001
    n_epochs = 4
    bs = 2
    grad_accum = 4
    weight_decay = 0.01
    if args.enable_loss_fn_weights:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([
            0.9, 1, 1.5, 0.8, 0.8, 0.8, 0.96, 1.1, 1.1, 3,  1, 1.1, 2, 3, 3,   2, 1, 2, 1, 2, 0.9, 0.75, 0.9, 0.75, 0.75, 0.7, 1, 2.5, 1, 0.75]).to(device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    awp_start_epoch = 1
    awp_lr = 1
    awp_eps = 0.001

    # Start training
    init_seed()
    folds = GroupKFold(n_splits=n_folds).split(
        X=train['question_body'], groups=train['question_body'])
    oofs = np.zeros((len(train), N_TARGETS))

    main_logger.info(f'Start training model {model_name}...')

    for fold_id, (train_index, valid_index) in enumerate(folds):

        main_logger.info(f'Fold {fold_id + 1} started at {time.ctime()}')

        fold_logger = init_logger(
            log_dir, f'train_fold_{fold_id+1}_{model_name}.log')

        train_loader = DataLoader(
            TextDataset(cat_features_train, ids_train['question'], ids_train['answer'],
                        seg_ids_train['question'], seg_ids_train['answer'], train_index, y),
            batch_size=bs, shuffle=True, num_workers=num_workers
        )
        valid_loader = DataLoader(
            TextDataset(cat_features_train, ids_train['question'], ids_train['answer'],
                        seg_ids_train['question'], seg_ids_train['answer'], valid_index, y),
            batch_size=bs, shuffle=False, num_workers=num_workers
        )

        model = models[model_name](
            use_attention_pooling=args.use_attention_pooling)

        optimizer = get_optimizer(model, lr, weight_decay, model_type)
        scheduler = OneCycleLR(optimizer, n_epochs=n_epochs,
                               n_batches=len(train_loader))

        # Initialize AWP if enabled
        awp = None
        if args.enable_awp:
            awp = AWP(
                model=model,
                optimizer=optimizer,
                adv_lr=awp_lr,
                adv_eps=awp_eps,
                start_epoch=awp_start_epoch,
                adv_param='weight'
            )
            main_logger.info(
                f'AWP enabled: lr={awp_lr}, eps={awp_eps}, start_epoch={awp_start_epoch}')
        learner = Learner(
            model,
            optimizer,
            train_loader,
            valid_loader,
            loss_fn,
            device,
            n_epochs,
            f'{model_name}_fold_{fold_id + 1}',
            checkpoint_dir,
            scheduler,
            metric_spec={'spearmanr': spearmanr_torch},
            monitor_metric=True,
            minimize_score=False,
            logger=fold_logger,
            grad_accum=grad_accum,
            batch_step_scheduler=True,
            awp=awp
        )
        learner.train()

        oofs[valid_index] = infer(
            learner.model, valid_loader, learner.best_checkpoint_file, device)

    main_logger.info(f'Finished training {model_name}')

    # Print CV scores
    ix = np.where(train.groupby("question_body")["host"].transform(
        "count") == 1)[0]  # unique question index
    main_logger.info('CVs:')
    main_logger.info(get_cvs(oofs, y, ix))

    # Store OOFs
    os.makedirs(oof_dir, exist_ok=True)
    pd.DataFrame(oofs, columns=TARGETS).to_csv(
        f'{oof_dir}{model_name}_tuned_oofs.csv')
