import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
from torch import Tensor
import torchtuples as tt
import argparse
from pycox.evaluation import EvalSurv
import warnings
import wandb
import pdb
import numpy as np
import pandas as pd
import warnings
from dsm import DeepSurvivalMachines
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival analysis configuration')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='kkbox_v1')
    parser.add_argument('--loss', type=str, default='rank')
    parser.add_argument('--optimizer', type=str, default='AdamWR')
    
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--distribution', type=str, default='Weibull')
    

    parser.add_argument('--num_nodes', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_BN', action='store_true')
    parser.add_argument('--use_output_bias', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    
    args = parser.parse_args()

    # print(args)


    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


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

    # patience=10 epochs
    max_epochs = 512
    patience = 10
    iter_per_epoch = train['x'].shape[0]/args.batch_size
    args.require_improvement = int(patience*iter_per_epoch)
    args.num_iterations = int(iter_per_epoch*max_epochs)

    # Model preparation =============================================================    
    layers = [args.num_nodes]* args.num_layers

    model = DeepSurvivalMachines(k=args.k,distribution=args.distribution,layers=layers,cuda=True, weight_decay=args.weight_decay, dropout=args.dropout)


    if args.wandb:
        wandb.init(project='ICLR_DSM_'+args.dataset+'_baseline', 
                group='DSM'+'_'+args.optimizer,
                name=f'L{args.num_layers}N{args.num_nodes}D{args.dropout}W{args.weight_decay}B{args.batch_size}',
                config=args)


    # Loss configuration ============================================================


    # Training ======================================================================
    model.fit(x_train, y_train[0], y_train[1], iters = args.num_iterations, learning_rate = args.lr,batch_size=args.batch_size)
    # log = model.fit(x_train, y_train_transformed, batch_size, epochs, callbacks, verbose, val_data = val_transformed, val_batch_size = batch_size)

    # Evaluation ===================================================================
        
    # _ = model.compute_baseline_hazards()
    surv = model.predict_survival(x_test.astype(np.double), np.sort(np.unique(y_train[0])).astype(np.double).tolist()).transpose()
    surv = pd.DataFrame(surv, index = np.sort(np.unique(y_train[0])), columns = [t for t in range(x_test.shape[0])])

    # surv = model.predict_surv_df(x_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    
    # ctd = ev.concordance_td()
    ctd = ev.concordance_td()
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

    ibs = ev.integrated_brier_score(time_grid)
    nbll = ev.integrated_nbll(time_grid)
    val_loss = model.compute_nll(x_val, y_val[0], y_val[1])

    import csv
    with open('./'+'ICLR_csv_'+args.dataset+'_'+'DSM.csv','a',newline='') as f:
        wr = csv.writer(f)

    if args.wandb:
        wandb.log({'val_loss':val_loss,
                    'ctd':ctd,
                    'ibs':ibs,
                    'nbll':nbll})
        wandb.finish()
    print(args)
    print(f"\n ctd: {ctd} \n ibs: {ibs} \n nbll: {nbll} \n val_loss: {val_loss}")


