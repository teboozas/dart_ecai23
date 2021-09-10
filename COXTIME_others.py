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
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime, MixedInputMLPCoxTime
import os
import random
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival analysis configuration')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='kkbox_v2')
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
    parser.add_argument('--start_iter', type=int, default=115)
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
    list_an=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    list_alpha=[0.0, 1e-4, 1e-5, 1e-6, 1e-7]
    list_lambda=[0.1, 0.01, 0.001, 0.0]
    list_log_durations= [True, False]

    # Training =====================================================================
    for fold in range(args.start_fold,args.end_fold):
        fold_ctd = []
        fold_ibs = []
        fold_nbll = []
        fold_val_loss = []
        start_iter = args.start_iter if (not args.start_fold==0)and(fold==args.start_fold) else 0
        
        for i in range(start_iter, 100):
            args.num_layers = random.choice(list_num_layers)
            args.num_nodes = random.choice(list_num_nodes)
            args.batch_size = random.choice(list_batch_size)
            args.lr = random.choice(list_lr) # not used
            args.weight_decay = random.choice(list_weight_decay)
            args.dropout = random.choice(list_dropout)
            args.an = random.choice(list_an)
            args.use_BN = True
            args.use_output_bias = False
            args.alpha=random.choice(list_alpha)
            args.beta =args.alpha
            args.log_duration = random.choice(list_log_durations)
            args.shrink = random.choice(list_lambda)
            
            print(f'[{fold} fold][{i+1}/100]')
            # print(args)

            df_train = data_split[str(fold)]['train']
            df_val = data_split[str(fold)]['valid']
            df_test = data_split[str(fold)]['test']
            if len(cols_categorical)>0:
                x_train = x_fit_transform(df_train)
                x_val = x_transform(df_val)
                x_test = x_transform(df_test)
            else:
                x_train = x_mapper.fit_transform(df_train).astype('float32')
                x_val = x_mapper.transform(df_val).astype('float32')
                x_test = x_mapper.transform(df_test).astype('float32')

            labtrans = CoxTime.label_transform(log_duration=args.log_duration)
            get_target = lambda df: (df['duration'].values, df['event'].values)
            y_train = labtrans.fit_transform(*get_target(df_train))
            y_val = labtrans.transform(*get_target(df_val))
            durations_test, events_test = get_target(df_test)
            val = tt.tuplefy(x_val, y_val)


            # Model preparation =============================================================    
            in_features = x_train[0].shape[1] if len(cols_categorical) else x_train.shape[1]
            num_nodes = [args.num_nodes]* args.num_layers
            out_features = 1
            batch_norm = args.use_BN
            dropout = args.dropout
            output_bias = args.use_output_bias
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if len(cols_categorical)>0:
                net = MixedInputMLPCoxTime(in_features, num_embeddings, embedding_dims, num_nodes, batch_norm, dropout)
                # net = Transformer(in_features, num_embeddings, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
            else:
                net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
            net = net.to(device)

            if args.optimizer == 'AdamWR':
                model = CoxTime(net,optimizer=tt.optim.AdamWR(lr=args.lr, decoupled_weight_decay=args.weight_decay,cycle_eta_multiplier=0.8),device=device, shrink=args.shrink, labtrans=labtrans)
            if args.wandb:
                wandb.init(project='ICLR_csv_'+args.dataset+'_baseline', 
                        group=f'coxtime_fold{fold}_'+args.loss,
                        name=f'L{args.num_layers}N{args.num_nodes}D{args.dropout}W{args.weight_decay}B{args.batch_size}',
                        config=args)

                wandb.watch(net)


            # Training ======================================================================
            batch_size = args.batch_size
            lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
            best = lrfinder.get_best_lr()

            model.optimizer.set_lr(best)
            
            epochs = args.epochs
            callbacks = [tt.callbacks.EarlyStopping()]
            verbose = True
            log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val.repeat(10).cat())
            # Evaluation ===================================================================
            
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(x_test)
            ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
            
            # ctd = ev.concordance_td()
            ctd = ev.concordance_td()
            time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

            ibs = ev.integrated_brier_score(time_grid)
            nbll = ev.integrated_nbll(time_grid)
            val_loss = min(log.monitors['val_'].scores['loss']['score'])

            import csv
            with open('./'+'ICLR_csv_'+args.dataset+'_'+'COXTIME.csv','a',newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['fold'+str(fold), i, val_loss, ctd, ibs, nbll, args])
            if args.wandb:
                wandb.log({'val_loss':val_loss,
                            'ctd':ctd,
                            'ibs':ibs,
                            'nbll':nbll})
                wandb.finish()