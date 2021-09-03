import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import random
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
from model import MixedInputMLP, Transformer, MLPVanilla
from loss_modified import DSAFTRankLoss,DSAFTMAELoss,DSAFTRMSELoss,DSAFTNKSPLLoss, DSAFTNKSPLLossNew
import os
import pickle

def get_score(n, t, y_test, delta_test, naf_base, kmf_cens, cens_test, exp_predict_neg_test, surv_residual, cens_residual, model):
    exp_residual_t = np.nan_to_num(np.exp(np.repeat(np.log(t),n) - model.predict(x_test).reshape(-1)))
    
    if surv_residual == True:
        H_base = naf_base.cumulative_hazard_at_times(exp_residual_t).values
    elif surv_residual == False:
        H_base = naf_base.cumulative_hazard_at_times(t).values        
    
    if cens_residual == True:
        cens_t = kmf_cens.survival_function_at_times(exp_residual_t).values
    elif cens_residual == False:
        cens_t = np.repeat(kmf_cens.survival_function_at_times(t).values, n)
    cens_t += 1e-16
    
    surv_cond = np.exp(-(H_base * exp_predict_neg_test)) - 1e-16
    
    indicator_first = (y_test <= t) * delta_test
    indicator_second = (y_test > t) * 1

    first_bs = np.power(surv_cond, 2) * indicator_first / cens_test
    second_bs = np.power(1 - surv_cond, 2) * indicator_second / cens_t
    bs = (first_bs + second_bs).sum() / n
    
    first_nbll = np.log(1 - surv_cond + 1e-16) * indicator_first / cens_test
    second_nbll = np.log(surv_cond + 1e-16) * indicator_second / cens_t
    nbll = (first_nbll + second_nbll).sum() / n
    
    return (bs, nbll)

def get_scores(model, y_test, delta_test, time_grid, surv_residual = False, cens_residual = False):
    n = y_test.shape[0]
    x_train, target = model.training_data
    y_train, delta_train = target

    # compute residual from training data
    exp_residual_train = np.nan_to_num(np.exp(np.log(y_train) - model.predict(x_train).reshape(-1)))
    exp_residual_test = np.nan_to_num(np.exp(np.log(y_test) - model.predict(x_test).reshape(-1)))

    # compute exp(-theta) from test data to evaluate accelerating component
    exp_predict_neg_test = np.nan_to_num(np.exp(-model.predict(x_test)).reshape(-1))

    naf_base = NelsonAalenFitter().fit(y_train, event_observed = delta_train)
    kmf_cens = KaplanMeierFitter().fit(y_train, event_observed = 1 - delta_train)
    
    if cens_residual == True:
        cens_test = kmf_cens.survival_function_at_times(exp_residual_test)
    elif cens_residual == False:
        cens_test = kmf_cens.survival_function_at_times(y_test)

    bss = []
    nblls = []
    for t in time_grid:
        bs, nbll = get_score(n, t, y_test, delta_test, naf_base, kmf_cens, cens_test, exp_predict_neg_test, surv_residual, cens_residual, model)
        bss.append(bs)
        nblls.append(-nbll)

    return (np.array(bss), np.array(nblls))


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
    
    args = parser.parse_args()

    print(args)


    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


   # Data preparation ==============================================================
    if args.dataset=='metabric':
        df_train = metabric.read_df()
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
        cols_categorical = []
    elif args.dataset=='gbsg':
        df_train = gbsg.read_df()
        cols_standardize = ["x3", "x4", "x5", "x6"]
        cols_leave = ["x0", "x2"]
        cols_categorical = ["x1"]
    elif args.dataset=='support':
        df_train = support.read_df()
        cols_standardize =  ["x0", "x3", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
        cols_leave = ["x1", "x4", "x5"]
        cols_categorical =  ["x2","x3", "x6"]
    elif args.dataset=='flchain':
        df_train = flchain.read_df()
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

    data_file_name = os.path.join('./data/',args.dataset+'.pickle')
    if os.path.exists(data_file_name):
        with open(data_file_name, 'rb') as f:
            data_split = pickle.load(f)
    else:
        data_fold = {}
        size = len(df_train)
        perm = list(range(size))
        random.shuffle(perm)
        for i in range(4):
            data_fold[str(i)] = df_train.loc[perm[i*int(size*0.2):(i+1)*int(size*0.2)]]
        data_fold['4'] = df_train.loc[perm[(i+1)*int(size*0.2):]]

        data_split = {}
        for i in range(4):
            data_split[str(i)] = {}
            data_split[str(i)]['test'] = data_fold[str(i)].copy()
            data_split[str(i)]['valid'] = data_fold[str(i+1)].copy()
            data_split[str(i)]['train'] = df_train.copy().drop(data_split[str(i)]['test'].index).drop(data_split[str(i)]['valid'].index)

        data_split['4'] = {}
        data_split['4']['test'] = data_fold[str(i+1)].copy()
        data_split['4']['valid'] = data_fold['0'].copy()
        data_split['4']['train'] = df_train.copy().drop(data_split['4']['test'].index).drop(data_split['4']['valid'].index)
        
        with open(data_file_name, 'wb') as f:
            pickle.dump(data_split, f, pickle.HIGHEST_PROTOCOL)

    # Hyperparameter setup =========================================================
    list_num_layers=[1, 2, 4]
    list_num_nodes=[64, 128, 256, 512]
    list_batch_size=[64, 128, 256, 512, 1024]
    list_lr=[1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    list_weight_decay=[0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0]
    list_dropout=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    list_an=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    list_alpha=[0.0, 1e-4, 1e-5, 1e-6, 1e-7]

    # Training =====================================================================
    FINAL_CTD = []
    FINAL_IBS = []
    FINAL_NBLL = []
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
            
            print(f'[{fold} fold][{i+1}/300]')
            # print(args)

            if len(cols_categorical)>0:
                x_train = x_fit_transform(data_split[str(fold)]['train'])
                x_val = x_transform(data_split[str(fold)]['valid'])
                x_test = x_transform(data_split[str(fold)]['test'])
            else:
                x_train = x_mapper.fit_transform(data_split[str(fold)]['train']).astype('float32')
                x_val = x_mapper.transform(data_split[str(fold)]['valid']).astype('float32')
                x_test = x_mapper.transform(data_split[str(fold)]['test']).astype('float32')


            get_target = lambda df: (df['duration'].values, df['event'].values)
            y_train = get_target(data_split[str(fold)]['train'])
            y_val = get_target(data_split[str(fold)]['valid'])
            durations_test, events_test = get_target(data_split[str(fold)]['test'])
            val = x_val, y_val

            def replace_zero(duration):
                return np.where(duration <= 0.0, duration + np.sort(np.unique(duration))[1], duration)

            # log-transformed time-to-event variables with replacement of zero-valued instances
            train_log_replace = np.log(replace_zero(y_train[0])).reshape(-1, 1)
            val_log_replace = np.log(replace_zero(y_val[0])).reshape(-1, 1)
            test_log_replace = np.log(replace_zero(durations_test)).reshape(-1, 1)

            # standardizer trained with training dataset
            scaler_train = StandardScaler().fit(train_log_replace)

            # scaled time-to-event datasets for consistent training
            y_train_transformed = (np.exp(scaler_train.transform(train_log_replace).reshape(-1)), y_train[1])
            y_val_transformed = (np.exp(scaler_train.transform(val_log_replace).reshape(-1)), y_val[1])
            val_transformed = x_val, y_val_transformed 
            durations_test_transformed = np.exp(scaler_train.transform(test_log_replace).reshape(-1))


            # Model preparation =============================================================    
            in_features = x_train[0].shape[1] if len(cols_categorical) else x_train.shape[1]
            num_nodes = [args.num_nodes]* args.num_layers
            out_features = 1
            batch_norm = args.use_BN
            dropout = args.dropout
            output_bias = args.use_output_bias
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if len(cols_categorical)>0:
                net = MixedInputMLP(in_features, num_embeddings, embedding_dims, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
                # net = Transformer(in_features, num_embeddings, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
            else:
                net = MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                            dropout, output_bias=output_bias)
            net = net.to(device)

            if args.optimizer == 'AdamWR':
                model = CoxPH(net, optimizer=tt.optim.AdamWR(lr=args.lr, decoupled_weight_decay=args.weight_decay),device=device)
            elif args.optimizer=='AdamW':
                model = CoxPH(net, optimizer=tt.optim.AdamW(lr=args.lr, decoupled_weight_decay=args.weight_decay),device=device)
            elif args.optimizer =='Adam':
                model = CoxPH(net, optimizer=tt.optim.Adam(lr=args.lr, weight_decay=args.weight_decay),device=device)
    

            wandb.init(project='icml_new_'+args.dataset, 
                    group=f'fold{fold}_'+args.loss+args.optimizer,
                    name=f'L{args.num_layers}N{args.num_nodes}D{args.dropout}W{args.weight_decay}B{args.batch_size}',
                    config=args)

            wandb.watch(net)

            # Loss configuration ============================================================

            patience=5
            if args.loss =='rank':
                model.loss = DSAFTRankLoss(alpha=args.alpha, beta=args.beta)
            elif args.loss == 'mae':
                model.loss = DSAFTMAELoss()
            elif args.loss == 'rmse':
                model.loss = DSAFTRMSELoss()
            elif args.loss =='kspl':
                model.loss = DSAFTNKSPLLoss(args.an, args.sigma)
            elif args.loss =='kspl_new':
                model.loss = DSAFTNKSPLLossNew(args.an, args.sigma)

            # Training ======================================================================
            batch_size = args.batch_size
            lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
            best = lrfinder.get_best_lr()

            model.optimizer.set_lr(best)
            
            epochs = args.epochs
            callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
            verbose = True
            log = model.fit(x_train, y_train_transformed, batch_size, epochs, callbacks, verbose, val_data = val_transformed, val_batch_size = batch_size)

            # Evaluation ===================================================================
            val_loss = min(log.monitors['val_'].scores['loss']['score'])
            
            # get Ctd
            ctd = concordance_index(event_times = durations_test_transformed,
                                    predicted_scores = model.predict(x_test).reshape(-1),
                                    event_observed = events_test)
            
            # set time grid for numerical integration to get IBS and IBLL
            if durations_test.min()>0:
                time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
            else:
                durations_test_copy = durations_test.copy()
                durations_test_copy.sort()
                time_grid = np.linspace(durations_test_copy[1], durations_test.max(), 100)
            # time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
            # transform time grid into DSAFT scale for fair comparison
            # pdb.set_trace()
            time_grid = np.exp(scaler_train.transform(np.log(time_grid.reshape(-1, 1)))).reshape(-1)
            # grid interval for numerical integration
            ds = np.array(time_grid - np.array([0.0] + time_grid[:-1].tolist()))
            # get BS's and NBLL's for given timepoints
            bs, nbll = get_scores(model, durations_test_transformed, events_test, time_grid, surv_residual = True, cens_residual = True)
            # get IBS
            ibs = sum(bs * ds) / (time_grid.max() - time_grid.min())
            ibll = sum(nbll * ds) / (time_grid.max() - time_grid.min())
            
            
            wandb.log({'val_loss':val_loss,
                        'ctd':ctd,
                        'ibs':ibs,
                        'nbll':ibll})
            wandb.finish()
            fold_ctd.append(ctd)
            fold_ibs.append(ibs)
            fold_nbll.append(nbll)
            fold_val_loss.append(val_loss)
        
        best_idx = np.array(val_loss).argmin()
        best_ctd = fold_ctd[best_idx]
        best_ibs = fold_ibs[best_idx]
        best_nbll = fold_nbll[best_idx]
        print(f'best_ctd:{best_ctd}/best_ibs:{best_ibs}/best_nbll:{best_nbll}')
        FINAL_CTD.append(best_ctd)
        FINAL_IBS.append(best_ibs)
        FINAL_NBLL.append(best_nbll)
    print('FINAL_CTD:',FINAL_CTD)
    print('FINAL_IBS:',FINAL_IBS)
    print('FINAL_NBLL:',FINAL_NBLL)
    print('AVG_CTD:',sum(FINAL_CTD)/5)
    print('AVG_IBS:',sum(FINAL_IBS)/5)
    print('AVG_NBLL:',sum(FINAL_NBLL)/5)