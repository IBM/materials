# Data
import pandas as pd
import numpy as np
import cupy as cp

# Machine learning
import optuna
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy import stats
from xgboost import XGBRegressor

import torch
import torch.nn.functional as F

# Standard library
import argparse


def main(args):
    print('Starting validation...')

    # load data and embeddings
    df_train = pd.read_csv(args.embeddings_path + 'train.csv')
    df_valid = pd.read_csv(args.embeddings_path + 'valid.csv')
    df_test = pd.read_csv(args.embeddings_path + 'test.csv')
    print(args.task_name)
    print('Data loaded.')

    # get data
    X_train = df_train.iloc[:, -1024:]
    X_valid = df_valid.iloc[:, -1024:]
    X_test = df_test.iloc[:, -1024:]
    y_train = df_train[args.task_name]
    y_valid = df_valid[args.task_name]
    y_test = df_test[args.task_name]

    # put to GPU
    X_train = cp.array(X_train)
    X_valid = cp.array(X_valid)
    X_test = cp.array(X_test)
    y_train = cp.array(y_train)
    y_valid = cp.array(y_valid)
    y_test = cp.array(y_test)

    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)

    def objective(trial):
        optimal_params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'gamma': trial.suggest_float('gamma', .0, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'random_state': args.seed,
        }
        clf = XGBRegressor(**optimal_params, device='gpu')  # xgboost >= 2.0
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        mae = mean_absolute_error(y_valid.get(), y_pred)
        return mae

    # optimize XGBoost
    print('Optimization...')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(best_params)

    torch.cuda.empty_cache()

    # train final XGBoost
    print('Training XGBoost...')
    xgb_predict = XGBRegressor(**best_params, device='gpu', random_state=args.seed)
    xgb_predict.fit(X_train, y_train)

    # get XGBoost predictions
    y_pred = xgb_predict.predict(X_test)
    y_test = y_test.get()

    # calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    spearman = stats.spearmanr(y_test, y_pred).correlation
    print(f'MAE: {mae:.4f}')
    print(f'R2: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'Spearman: {spearman:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--embeddings_path', type=str, default='../data/embeddings/clip_embeddings_qm9_', required=True)
    parser.add_argument('--seed', type=int, default=0, required=False)
    args = parser.parse_args()
    main(args)