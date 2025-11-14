# Data
import pandas as pd
import numpy as np
import cupy as cp

# Machine learning
import optuna
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import stats
from xgboost import XGBRegressor

# Standard library
import time
import argparse


def main(args):
    print('Starting validation...')
    measure_name = args.task
    random_state = 0

    # load data and embeddings
    df_train = pd.read_csv(f'../inference/vqgan_embeddings_epoch=43_finetune_qm9_{measure_name}_train.csv')
    df_valid = pd.read_csv(f'../inference/vqgan_embeddings_epoch=43_finetune_qm9_{measure_name}_valid.csv')
    df_test = pd.read_csv(f'../inference/vqgan_embeddings_epoch=43_finetune_qm9_{measure_name}_test.csv')
    print(measure_name)
    print('Data loaded.')

    # get data
    X_train = df_train.iloc[:, :2048]
    X_valid = df_valid.iloc[:, :2048]
    X_test = df_test.iloc[:, :2048]
    y_train = df_train[measure_name]
    y_valid = df_valid[measure_name]
    y_test = df_test[measure_name]

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
            'objective': 'reg:squaredlogerror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'gamma': trial.suggest_float('gamma', .0, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'random_state': random_state,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
        # clf = XGBRegressor(**optimal_params, device='gpu', random_state=random_state)  # xgboost >= 2.0
        clf = XGBRegressor(**optimal_params)  # xgboost < 2.0
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        mae = mean_absolute_error(y_valid.get(), y_pred)
        return mae

    # optimize XGBoost
    print('Optimization...')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=500)
    best_params = study.best_params
    print(best_params)

    # train final XGBoost
    print('Training XGBoost...')
    tic = time.time()
    # xgb_predict = XGBRegressor(**best_params, device='gpu')
    xgb_predict = XGBRegressor(**best_params, tree_method='gpu_hist', gpu_id=0, random_state=random_state)
    xgb_predict.fit(X_train, y_train)
    toc = time.time()
    print('ETA:', toc-tic)

    # get XGBoost predictions
    y_pred = xgb_predict.predict(X_test)
    y_test = y_test.get()

    # calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    spearman = stats.spearmanr(y_test, y_pred).correlation
    print(f'MAE: {mae:.4f}')
    print(f'R2: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'Spearman: {spearman:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()
    main(args)