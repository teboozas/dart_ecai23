import warnings
import math
import os
import random
import pickle
import argparse
import wandb
import pdb
import easydict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

# from pycox.datasets import metabric, gbsg, support, flchain#, nwtco
from pycox.models import CoxPH, CoxCC
from pycox.evaluation import EvalSurv
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
#from model import MixedInputMLP
import torch
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index

# Tensorflow verision required == 1.15
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.simplefilter("ignore")

from draft.model.date import DATE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival analysis configuration')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dataset', type=str, default='kkbox_v2')#kkbox_v2

    parser.add_argument('--num_nodes', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=512)
    
    # args for DRAFT model
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--require_improvement', type=int, default=100)
    parser.add_argument('--num_iterations', type=int, default=9999)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--latent_dim', type=int, default=50)

    # args for DATE only
    parser.add_argument('--gen_updates', type=int, default=2)
    parser.add_argument('--disc_updates', type=int, default=1)

    # python DATE_others.py --wandb --dataset=gbsg --start_fold=0 --end_fold=1
    # source activate pycox
    parser.add_argument('--start_fold', type=int, default=0)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--end_fold', type=int, default=5)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    print(args)

    # GPUID = int(tf.test.is_gpu_available())
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # Data preparation ==============================================================

    if args.dataset=='kkbox_v1':
        kkbox_v1 = pickle.load(open('./data/kkbox_v1.pickle','rb'))
        df_train = pd.concat([kkbox_v1['train'],kkbox_v1['val'],kkbox_v1['test'] ]) 
        end_t = df_train['duration'].max()
        covariates = list(df_train.columns)        

        imputation_values = []
        imputation_values_dict = {}
        # city, gender, and registered_via contain nan values 
        for k in df_train.keys():
            if k =='msno' or k=='gender':
                imputation_values.append(0)
            else:
                imputation_values.append(np.nanmedian(df_train[k], axis = 0))
                if df_train[k].isnull().sum()>0:
                    imputation_values_dict[k] = imputation_values[-1]

        df_train = kkbox_v1['train']
        df_val = kkbox_v1['val']
        df_test = kkbox_v1['test']
        

        cols_standardize = ['n_prev_churns', 'log_days_between_subs', 'log_days_since_reg_init' ,'age_at_start', 'log_payment_plan_days', 'log_plan_list_price', 'log_actual_amount_paid']
        cols_leave =['is_auto_renew', 'is_cancel', 'strange_age', 'nan_days_since_reg_init', 'no_prev_churns']
        cols_categorical = ['city', 'gender', 'registered_via']
    
    elif args.dataset=='kkbox_v2':
        df_train = pickle.load(open('./data/kkbox_v2.pickle','rb'))
        end_t = df_train['duration'].max()
        covariates = list(df_train.columns)
        
        imputation_values = []
        imputation_values_dict = {}
        # city, gender, and registered_via contain nan values 
        for k in df_train.keys():
            if k =='msno' or k=='gender':
                imputation_values.append(0)
            else:
                imputation_values.append(np.nanmedian(df_train[k], axis = 0))
                if df_train[k].isnull().sum()>0:
                    imputation_values_dict[k] = imputation_values[-1]


        cols_standardize = ['n_prev_churns', 'log_days_between_subs', 'log_days_since_reg_init' ,'age_at_start', 'log_payment_plan_days', 'log_plan_list_price', 'log_actual_amount_paid']
        cols_leave =['is_auto_renew', 'is_cancel', 'strange_age', 'nan_days_since_reg_init', 'no_prev_churns']
        cols_categorical = ['city', 'gender', 'registered_via','payment_method_id']
    else:
        print("#"*30)
        print("dataset should be \{ kkbox_v1,kkbox_v2 \}")
        print("#"*30)
        assert False

    if args.dataset == 'kkbox_v2':
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)

    # imputation
    df_train['gender'] = df_train['gender'].astype('category').cat.codes+1
    df_val['gender'] = df_val['gender'].astype('category').cat.codes+1
    df_test['gender'] = df_test['gender'].astype('category').cat.codes+1
    for k in imputation_values_dict.keys():
        df_train[k] = df_train[k].fillna(imputation_values_dict[k])
        df_val[k] = df_val[k].fillna(imputation_values_dict[k])
        df_test[k] = df_test[k].fillna(imputation_values_dict[k])
    

    split_train = df_train
    split_valid = df_val
    split_test = df_test

    one_hot_columns = []
    one_hot_indices = []
    cols_leave_modified = []

    for col in cols_categorical:
        tmp_train = pd.get_dummies(split_train[col], prefix = col)
        tmp_valid = pd.get_dummies(split_valid[col], prefix = col)
        tmp_test = pd.get_dummies(split_test[col], prefix = col)
        total_keys = set(list(tmp_train.keys()) + list(tmp_valid.keys()) + list(tmp_test.keys()))
        for k in total_keys:
            if k not in list(tmp_train.keys()):
                tmp_dummies = np.zeros(tmp_train.shape[0]).astype('uint8')
                tmp_train[k] = tmp_dummies
            if k not in list(tmp_valid.keys()):
                tmp_dummies = np.zeros(tmp_valid.shape[0]).astype('uint8')
                tmp_valid[k] = tmp_dummies
            if k not in list(tmp_test.keys()):
                tmp_dummies = np.zeros(tmp_test.shape[0]).astype('uint8')
                tmp_test[k] = tmp_dummies

        tmp_train = tmp_train.sort_index(axis=1)
        tmp_valid = tmp_valid.sort_index(axis=1)
        tmp_test = tmp_test.sort_index(axis=1)
        one_hot_columns.append(list(tmp_train.columns))
        split_train = pd.concat([tmp_train, split_train], axis = 1)
        split_valid = pd.concat([tmp_valid, split_valid], axis = 1)
        split_test = pd.concat([tmp_test, split_test], axis = 1)
        split_train.drop(col, axis = 1)
        split_valid.drop(col, axis = 1)
        split_test.drop(col, axis = 1)

    one_hot_columns.reverse()

    i = 0

    for cols in one_hot_columns:
        tmp = []
        for j in range(len(cols)):
            tmp.append(i)
            i += 1
        one_hot_indices.append(tmp)
        cols_leave_modified.extend(cols)
        
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave_modified]
    x_mapper = DataFrameMapper(leave + standardize)

    x_train = x_mapper.fit_transform(split_train).astype('float32')
    x_val = x_mapper.transform(split_valid).astype('float32')
    x_test = x_mapper.transform(split_test).astype('float32')

    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(split_train)
    y_val = get_target(split_valid)
    durations_test, events_test = get_target(split_test)

    train = {'x': x_train, 'e': y_train[1], 't': y_train[0]}
    valid = {'x': x_val, 'e': y_val[1], 't': y_val[0]}
    test = {'x': x_test, 'e': events_test, 't': durations_test}

    
    # patience=10 epochs
    max_epochs = 512
    iter_per_epoch = train['x'].shape[0]/args.batch_size
    args.require_improvement = int(10*iter_per_epoch)
    args.num_iterations = int(iter_per_epoch*max_epochs)

    model = DATE(learning_rate = args.lr,
                require_improvement = args.require_improvement,
                num_iterations = args.num_iterations,
                batch_size = args.batch_size,
                beta1 = args.beta1,
                beta2 = args.beta2,
                l2_reg = args.l2_reg,
                keep_prob = args.keep_prob,
                latent_dim = args.latent_dim,
                gen_updates = args.gen_updates,
                disc_updates = args.disc_updates,
                hidden_dim = np.repeat(args.num_nodes, args.num_layers).tolist(),
                train_data = train,
                valid_data = valid,
                test_data = test,
                input_dim = train['x'].shape[1],
                num_examples = train['x'].shape[0],
                covariates = covariates,
                categorical_indices = one_hot_indices,
                imputation_values = imputation_values,
                end_t = end_t,
                seed = args.seed,
                sample_size = 100,
                max_epochs = 1000,
                path_large_data = "")

    # Training ======================================================================
    if args.wandb:
        wandb.init(project='ICLR_'+args.dataset+"_baseline", 
                group=f"DATE",
                name=f'LR{args.lr}_L2{args.l2_reg}_DIM{args.num_nodes}_L{args.num_layers}',
                config=args)

    # wandb.watch(model)

    with model.session:
        model.train_test()


    if args.wandb:
        wandb.log({'val_loss':model.val_loss,
                'ctd':model.ctd,
                'ibs':model.ibs,
                'nbll':model.nbll})
        wandb.finish()