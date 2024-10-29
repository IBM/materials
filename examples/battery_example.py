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

train_df  = pd.read_csv(f"../data/lce/train.csv").dropna()
test_df  = pd.read_csv(f"../data/lce/test.csv").dropna()


# Make a list of smiles
train_smiles_list = pd.concat([train_df[f'smi{i}'] for i in range(1, 7)]).unique().tolist()
test_smiles_list = pd.concat([test_df[f'smi{i}'] for i in range(1, 7)]).unique().tolist()

fm4m.avail_models()

model_type = "SELFIES-TED"
train_emb, test_emb = fm4m.get_representation(train_smiles_list,test_smiles_list, model_type, return_tensor=False)

train_emb = [np.nan if row.isna().all() else row.dropna().tolist() for _, row in train_emb.iterrows()]
test_emb = [np.nan if row.isna().all() else row.dropna().tolist() for _, row in test_emb.iterrows()]

train_dict = dict(zip(train_smiles_list, train_emb))
test_dict = dict(zip(test_smiles_list, test_emb))

def replace_with_list(value, my_dict):
    return my_dict.get(value, value)

# Replacement the smiles string with its embeddings
df_train_emb = train_df.applymap(lambda x: replace_with_list(x, train_dict))
df_test_emb = test_df.applymap(lambda x: replace_with_list(x, test_dict))

# Drop rows with NaN and reset index
df_train_emb = df_train_emb.dropna().reset_index(drop=True)
df_test_emb = df_test_emb.dropna().reset_index(drop=True)

# Define a function to handle repetitive tasks
def compute_components(df, smi_cols, conc_cols):
    components = [df[smi].apply(pd.Series).mul(df[conc], axis=0) for smi, conc in zip(smi_cols, conc_cols)]
    return sum(components)

# List of columns to process
smi_cols = [f'smi{i}' for i in range(1, 7)]
conc_cols = [f'conc{i}' for i in range(1, 7)]

# Train data processing
x_train = compute_components(df_train_emb, smi_cols, conc_cols)
y_train = pd.DataFrame(df_train_emb["LCE"], columns=["LCE"])

# Test data processing
X_test = compute_components(df_test_emb, smi_cols, conc_cols)
y_test = pd.DataFrame(df_test_emb["LCE"], columns=["LCE"])

regressor = SVR(kernel="rbf", degree=3, C=5, gamma="scale", epsilon=0.01)
model = TransformedTargetRegressor(regressor=regressor,
                                   transformer=MinMaxScaler(feature_range=(-1, 1))
                                   ).fit(x_train, y_train)

y_prob = model.predict(X_test)
RMSE_score = mean_squared_error(y_test, y_prob, squared=False)
print(RMSE_score)

