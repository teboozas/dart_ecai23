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
import wandb
import pdb
import numpy as np
import pandas as pd
import warnings
import math
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong

from dsm import DeepSurvivalMachines

import os
import random
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival analysis configuration')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='metabric')
    parser.add_argument('--loss', type=str, default='rank')
    parser.add_argument('--optimizer', type=str, default='AdamWR')
    parser.add_argument('--an', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    ## alpha, beta: penalty size for Rank loss
    ## default search space: 1e-4 ~ 1e-8
    ## let alpha == beta at first
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--start_fold', type=int, default=0)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--end_fold', type=int, default=5)
    

    parser.add_argument('--num_nodes', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
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

    data_file_name = os.path.join('./data/',args.dataset+'.pickle')
    if os.path.exists(data_file_name):
        with open(data_file_name, 'rb') as f:
            data_split = pickle.load(f)
    df_train = pd.concat([data_split['0']['train'],data_split['0']['valid'],data_split['0']['test'] ]) 

   # Data preparation ==============================================================
    if args.dataset=='metabric':
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
        cols_categorical = []
    elif args.dataset=='gbsg':
        cols_standardize = ["x3", "x4", "x5", "x6"]
        cols_leave = ["x0", "x2"]
        cols_categorical = ["x1"]
    elif args.dataset=='support':
        cols_standardize =  ["x0", "x3", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
        cols_leave = ["x1", "x4", "x5"]
        cols_categorical =  ["x2","x3", "x6"]
    elif args.dataset=='flchain':
        df_train.columns =  ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "duration", "event"]
        cols_standardize =  ["x0", "x3", "x4", "x6"]
        cols_leave = ["x1", "x7"]
        cols_categorical = ["x2", "x5"]

    if len(cols_categorical)>0:
        num_embeddings = [len(df_train[cat].unique())+1 for cat in cols_categorical]
        embedding_dims = [math.ceil(n_emb/2) for n_emb in num_embeddings]
            
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

        x_mapper_float = DataFrameMapper(standardize + leave)
        x_mapper_long = DataFrameMapper(categorical)
        x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df).astype(np.float32), x_mapper_long.fit_transform(df))
        x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df).astype(np.float32), x_mapper_long.transform(df))
    else:        
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)

    # Hyperparameter setup =========================================================
    list_num_layers=[1, 2, 4]
    list_num_nodes=[64, 128, 256, 512]
    list_batch_size=[64, 128, 256, 512, 1024]
    list_lr=[1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    list_weight_decay=[0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001]
    list_dropout=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    list_distribution = ['LogNormal', 'Weibull']#,'Normal']
    list_k = [3, 4,5, 6]

    # Training =====================================================================
    # pdb.set_trace()
    for fold in range(args.start_fold,args.end_fold):
        fold_ctd = []
        fold_ibs = []
        fold_nbll = []
        fold_val_loss = []
        start_iter = args.start_iter if (not args.start_fold==0)and(fold==args.start_fold) else 0
        
        split_train = data_split[str(fold)]['train']
        split_valid = data_split[str(fold)]['valid']
        split_test = data_split[str(fold)]['test']

        for i in range(start_iter, 300):
            args.num_layers = random.choice(list_num_layers)
            args.num_nodes = random.choice(list_num_nodes)
            args.batch_size = random.choice(list_batch_size)
            args.lr = random.choice(list_lr) # not used
            args.weight_decay = random.choice(list_weight_decay)
            args.dropout = random.choice(list_dropout)
            args.distribution = random.choice(list_distribution)
            args.k = random.choice(list_k)

            args.use_BN = True
            args.use_output_bias = False
            layers = [args.num_nodes]* args.num_layers
            
            # print(f'[{fold} fold][{i+1}/100]')
            # print(args)

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
            

            # Model preparation =============================================================    
            # in_features = x_train[0].shape[1] if len(cols_categorical) else x_train.shape[1]
            num_nodes = [args.num_nodes]* args.num_layers
            out_features = 1
            batch_norm = args.use_BN
            dropout = args.dropout
            output_bias = args.use_output_bias
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # pdb.set_trace()
            model = DeepSurvivalMachines(k=args.k,distribution=args.distribution,layers=layers,cuda=True, weight_decay=args.weight_decay, dropout=args.dropout)

            if args.wandb:
                wandb.init(project='ICLR_DSM_'+args.dataset+'_baseline', 
                        group=f'DSM_fold{fold}',
                        name=f'L{args.num_layers}N{args.num_nodes}D{args.dropout}W{args.weight_decay}B{args.batch_size}',
                        config=args)

                # wandb.watch(model)


            max_epochs = args.epochs
            # patience = 10
            # pdb.set_trace()
            iter_per_epoch = x_train.shape[0]/args.batch_size
            # args.require_improvement = int(patience*iter_per_epoch)
            args.num_iterations = int(iter_per_epoch*max_epochs)
            # Training ======================================================================

            # pdb.set_trace()
            model.fit(x_train, y_train[0], y_train[1], iters = args.num_iterations, learning_rate = args.lr,batch_size=args.batch_size)
            # pdb.set_trace()
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
                wr.writerow(['fold'+str(fold), i, val_loss, ctd, ibs, nbll, args])

            if args.wandb:
                wandb.log({'val_loss':val_loss,
                            'ctd':ctd,
                            'ibs':ibs,
                            'nbll':nbll})
                wandb.finish()
            print(args)
            print(f"\n ctd: {ctd} \n ibs: {ibs} \n nbll: {nbll} \n val_loss: {val_loss}")

            break
        break