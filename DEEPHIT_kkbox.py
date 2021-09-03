import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
from torch import Tensor
import torchtuples as tt
import argparse
from pycox.datasets import metabric, gbsg, support, flchain, nwtco
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import warnings
from lifelines import KaplanMeierFitter
import wandb
import pdb
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
import warnings
import math
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from model import MixedInputMLP, Transformer
from loss import DSAFTRankLoss,DSAFTMAELoss,DSAFTRMSELoss,DSAFTNKSPLLoss, DSAFTNKSPLLossNew

from pycox.models import DeepHitSingle
from pycox.models.cox_time import MLPVanillaCoxTime, MixedInputMLPCoxTime
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival analysis configuration')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='kkbox')
    parser.add_argument('--loss', type=str, default='rank')
    parser.add_argument('--optimizer', type=str, default='AdamWR')
    parser.add_argument('--an', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    

    parser.add_argument('--num_nodes', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_BN', action='store_true')
    parser.add_argument('--use_output_bias', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    
    args = parser.parse_args()

    print(args)


    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

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



    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

    x_mapper_float = DataFrameMapper(standardize + leave)
    x_mapper_long = DataFrameMapper(categorical)


    x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df).astype(np.float32), x_mapper_long.fit_transform(df))
    x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df).astype(np.float32), x_mapper_long.transform(df))

    x_train = x_fit_transform(df_train)
    x_val = x_transform(df_val)
    x_test = x_transform(df_test)
    num_embeddings = x_train[1].max(0) + 1
    embedding_dims = num_embeddings // 2
    
    num_durations = 10
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    train = (x_train, y_train)
    val = (x_val, y_val)

    # We don't need to transform the test labels
    durations_test, events_test = get_target(df_test)


    # Model preparation =============================================================    
    in_features = x_train[0].shape[1]
    num_nodes = [args.num_nodes]* args.num_layers
    out_features = labtrans.out_features
    batch_norm = args.use_BN
    dropout = args.dropout
    output_bias = args.use_output_bias
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = args.batch_size
    # net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
    #                             dropout, output_bias=output_bias)
    net = MixedInputMLP(in_features, num_embeddings, embedding_dims, num_nodes, out_features, batch_norm=batch_norm, dropout=dropout, output_bias=output_bias)
    # net = Transformer(in_features, num_embeddings, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
    # net = MixedInputMLPCoxTime(in_features, num_embeddings, embedding_dims, num_nodes, batch_norm, dropout)
    net = net.to(device)

    if args.optimizer == 'AdamWR':
        model = DeepHitSingle(net,optimizer=tt.optim.AdamWR(lr=args.lr, decoupled_weight_decay=args.weight_decay,cycle_eta_multiplier=0.8), alpha=0.001, sigma=0.5, duration_index=labtrans.cuts)
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=3)
    lr = lr_finder.get_best_lr()
    model.optimizer.set_lr(lr)
    # model.optimizer.set_lr(1e-5)

    if args.wandb:
        wandb.init(project='ICLR_'+args.dataset+'_baseline', #'lr_test',#
                group='deephit'+'_'+args.optimizer,
                name=f'L{args.num_layers}N{args.num_nodes}D{args.dropout}W{args.weight_decay}B{args.batch_size}',
                config=args)

        wandb.watch(net)

    # Loss configuration ============================================================


    # Training ======================================================================

    epochs = args.epochs
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
    # log = model.fit(x_train, y_train_transformed, batch_size, epochs, callbacks, verbose, val_data = val_transformed, val_batch_size = batch_size)

    # Evaluation ===================================================================
    surv = model.interpolate(10).predict_surv_df(x_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    
    # ctd = ev.concordance_td()
    ctd = ev.concordance_td('antolini')
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

    ibs = ev.integrated_brier_score(time_grid)
    nbll = ev.integrated_nbll(time_grid)
    val_loss = min(log.monitors['val_'].scores['loss']['score'])

    if args.wandb:
        wandb.log({'val_loss':val_loss,
                    'ctd':ctd,
                    'ibs':ibs,
                    'nbll':nbll})
        wandb.finish()

