import sys
sys.path.append("../models")
sys.path.append("../")

import models.fm4m as fm4m
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_df  = pd.read_csv(f"../data/bace/train.csv")
test_df  = pd.read_csv(f"../data/bace/test.csv")

input = "smiles"
output = "Class"
task_name = "bace"

xtrain = list(train_df[input].values)
ytrain = list(train_df[output].values)

xtest = list(test_df[input].values)
ytest = list(test_df[output].values)

model_type = "SELFIES-TED"
x_batch, x_batch_test = fm4m.get_representation(xtrain, xtest, model_type = model_type, return_tensor = False)
result = fm4m.single_modal(model=model_type, task_name=task_name, x_train=xtrain, y_train=ytrain, x_test=xtest, y_test=ytest, downstream_model="XGBClassifier", save_model=True)

