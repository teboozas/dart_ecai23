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
    parser.add_argument('--dataset', type=str, default='metabric')#kkbox_v2

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

    # python search_modified_date.py --wandb --dataset=flchain --start_fold=0 --end_fold=1
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


    data_file_name = os.path.join('./data/',args.dataset+'.pickle')
    if os.path.exists(data_file_name):
        with open(data_file_name, 'rb') as f:
            data_split = pickle.load(f)
    df_train = pd.concat([data_split['0']['train'],data_split['0']['valid'],data_split['0']['test'] ]) 

    # Data preparation ==============================================================
    if args.dataset=='metabric':
        # df_train = metabric.read_df()
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_categorical = ['x4', 'x5', 'x6', 'x7']
    elif args.dataset=='gbsg':
        # df_train = gbsg.read_df()
        cols_standardize = ["x3", "x4", "x5", "x6"]
        cols_categorical = ["x0", "x1" , "x2"]
    elif args.dataset=='support':
        # df_train = support.read_df()
        cols_standardize =  ["x0", "x3", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
        cols_categorical = ["x1", "x2", "x4", "x5", "x6"]
    elif args.dataset=='flchain':
        # df_train = flchain.read_df()
        df_train.columns =  ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "duration", "event"]
        cols_standardize =  ["x0", "x3", "x4", "x6"]
        cols_categorical = ["x1", "x2", "x5", "x7"]
    elif args.dataset=='kkbox':
        from kkbox import _DatasetKKBoxChurn
        kkbox_v1 = _DatasetKKBoxChurn()
        try:
            df_train = kkbox_v1.read_df(subset='train')
            df_val = kkbox_v1.read_df(subset='val')
            df_test = kkbox_v1.read_df(subset='test')
        except:
            kkbox_v1.download_kkbox()
            df_train = kkbox_v1.read_df(subset='train')
            df_val = kkbox_v1.read_df(subset='val')
            df_test = kkbox_v1.read_df(subset='test')
        cols_standardize = ['n_prev_churns', 'log_days_between_subs', 'log_days_since_reg_init' ,'age_at_start', 'log_payment_plan_days', 'log_plan_list_price', 'log_actual_amount_paid']
        cols_categorical =['city', 'gender', 'registered_via', 'is_auto_renew', 'is_cancel', 'strange_age', 'nan_days_since_reg_init', 'no_prev_churns']

    elif args.dataset=='kkbox_v2':
        from kkbox import _DatasetKKBoxAdmin
        kkbox_v2 = _DatasetKKBoxAdmin()
        try:
            df_train = kkbox_v2.read_df()
        except:
            kkbox_v2.download_kkbox()
            df_train = kkbox_v2.read_df()
        cols_standardize = ['n_prev_churns', 'log_days_between_subs', 'log_days_since_reg_init' ,'age_at_start', 'log_payment_plan_days', 'log_plan_list_price', 'log_actual_amount_paid']
        cols_categorical =['city', 'gender', 'registered_via','payment_method_id', 'is_auto_renew', 'is_cancel', 'strange_age', 'nan_days_since_reg_init', 'no_prev_churns']

    end_t = df_train['duration'].max()
    covariates = list(df_train.columns)
    imputation_values = np.nanmedian(df_train, axis = 0)
    # pdb.set_trace()




    # Data Split ======================================================
    # if args.dataset=='kkbox_v2':
    #     df_test = df_train.sample(frac=0.25)
    #     df_train = df_train.drop(df_test.index)
    #     df_val = df_train.sample(frac=0.1)
    #     df_train = df_train.drop(df_val.index)
    # elif not (args.dataset == 'kkbox'):
    #     df_test = df_train.sample(frac=0.2)
    #     df_train = df_train.drop(df_test.index)
    #     df_val = df_train.sample(frac=0.2)
    #     df_train = df_train.drop(df_val.index)

    # pdb.set_trace()



    # Hyperparameter setup =========================================================
    list_lr=[1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    list_batch_size=[64, 128, 256, 512, 1024]
    list_weight_decay=[0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001] # l2_reg 
    list_dropout=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # 1-keep_prob
    list_num_nodes=[64, 128, 256, 512] # same with latent_dim
    list_num_layers=[1, 2, 4]
    

    FINAL_CTD = []
    FINAL_IBS = []
    FINAL_NBLL = []
    for fold in range(args.start_fold,args.end_fold):

        fold_ctd = []
        fold_ibs = []
        fold_nbll = []
        fold_val_loss = []
        start_iter = args.start_iter if (not args.start_fold==0)and(fold==args.start_fold) else 0
        
        split_train = data_split[str(fold)]['train']
        split_valid = data_split[str(fold)]['valid']
        split_test = data_split[str(fold)]['test']

        for i in range(start_iter,300):
            
            print(f'[{fold} fold][{i+1}/300]')

            args.lr = random.choice(list_lr)
            args.batch_size = random.choice(list_batch_size)
            args.l2_reg = random.choice(list_weight_decay)
            args.keep_prob = 1-random.choice(list_dropout)
            args.num_nodes = random.choice(list_num_nodes)
            args.latent_dim = args.num_nodes
            args.num_layers = random.choice(list_num_layers)

            print(args)

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
                wandb.init(project='aaai_'+args.dataset+"_DATE&DRAFT", 
                        group=f"DATE_fold{fold}",
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