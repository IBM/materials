from sklearn.metrics import roc_auc_score, roc_curve

import datetime
import os
import umap
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json

from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_squared_error
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
import json
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler


import torch
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.append("models/")

from models.selfies_ted.load import SELFIES as bart
from models.mhg_model import load as mhg
from models.smi_ted.smi_ted_light.load import load_smi_ted

import mordred
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

datasets = {}
models = {}
downstream_models ={}


def avail_models_data():
    global datasets
    global models

    datasets = [{"Dataset": "hiv", "Input": "smiles", "Output": "HIV_active", "Path": "data/hiv", "Timestamp": "2024-06-26 11:27:37"},
  {"Dataset": "esol", "Input": "smiles", "Output": "ESOL predicted log solubility in mols per litre", "Path": "data/esol", "Timestamp": "2024-06-26 11:31:46"},
  {"Dataset": "freesolv", "Input": "smiles", "Output": "expt", "Path": "data/freesolv", "Timestamp": "2024-06-26 11:33:47"},
  {"Dataset": "lipo", "Input": "smiles", "Output": "y", "Path": "data/lipo", "Timestamp": "2024-06-26 11:34:37"},
  {"Dataset": "bace", "Input": "smiles", "Output": "Class", "Path": "data/bace", "Timestamp": "2024-06-26 11:36:40"},
  {"Dataset": "bbbp", "Input": "smiles", "Output": "p_np", "Path": "data/bbbp", "Timestamp": "2024-06-26 11:39:23"},
  {"Dataset": "clintox", "Input": "smiles", "Output": "CT_TOX", "Path": "data/clintox", "Timestamp": "2024-06-26 11:42:43"}]


    models = [{"Name": "bart","Model Name": "SELFIES-TED","Description": "BART model for string based SELFIES modality", "Timestamp": "2024-06-21 12:32:20"},
  {"Name": "mol-xl","Model Name": "MolFormer", "Description": "MolFormer model for string based SMILES modality", "Timestamp": "2024-06-21 12:35:56"},
  {"Name": "mhg", "Model Name": "MHG-GED","Description": "Molecular hypergraph model", "Timestamp": "2024-07-10 00:09:42"},
  {"Name": "smi-ted", "Model Name": "SMI-TED","Description": "SMILES based encoder decoder model", "Timestamp": "2024-07-10 00:09:42"}]


def avail_models(raw=False):
    global models

    models = [{"Name": "smi-ted", "Model Name": "SMI-TED","Description": "SMILES based encoder decoder model"},
              {"Name": "bart","Model Name": "SELFIES-TED","Description": "BART model for string based SELFIES modality"},
              {"Name": "mol-xl","Model Name": "MolFormer", "Description": "MolFormer model for string based SMILES modality"},
              {"Name": "mhg", "Model Name": "MHG-GED","Description": "Molecular hypergraph model"},
              {"Name": "Mordred", "Model Name": "Mordred","Description": "A molecular descriptor calculator"},
              {"Name": "MorganFingerprint", "Model Name": "MorganFingerprint","Description": "Encodes molecular structures into binary vectors based on circular atom environments"}              
  ]



    if raw: return models
    else:
        return pd.DataFrame(models).drop('Name', axis=1)

    return models

def avail_downstream_models(raw=False):
    global downstream_models

    downstream_models = [{"Name": "XGBClassifier", "Task Type": "Classfication"},
                         {"Name": "DefaultClassifier", "Task Type": "Classfication"},
                        {"Name": "SVR", "Task Type": "Regression"},
                        {"Name": "Kernel Ridge", "Task Type": "Regression"},
                        {"Name": "Linear Regression", "Task Type": "Regression"},
                        {"Name": "DefaultRegressor", "Task Type": "Regression"},
                         ]

    if raw: return downstream_models
    else:
        return pd.DataFrame(downstream_models)



def avail_datasets():
    global datasets

    datasets = [{"Dataset": "hiv", "Input": "smiles", "Output": "HIV_active", "Path": "data/hiv",
                 "Timestamp": "2024-06-26 11:27:37"},
                {"Dataset": "esol", "Input": "smiles", "Output": "ESOL predicted log solubility in mols per litre",
                 "Path": "data/esol", "Timestamp": "2024-06-26 11:31:46"},
                {"Dataset": "freesolv", "Input": "smiles", "Output": "expt", "Path": "data/freesolv",
                 "Timestamp": "2024-06-26 11:33:47"},
                {"Dataset": "lipo", "Input": "smiles", "Output": "y", "Path": "data/lipo",
                 "Timestamp": "2024-06-26 11:34:37"},
                {"Dataset": "bace", "Input": "smiles", "Output": "Class", "Path": "data/bace",
                 "Timestamp": "2024-06-26 11:36:40"},
                {"Dataset": "bbbp", "Input": "smiles", "Output": "p_np", "Path": "data/bbbp",
                 "Timestamp": "2024-06-26 11:39:23"},
                {"Dataset": "clintox", "Input": "smiles", "Output": "CT_TOX", "Path": "data/clintox",
                 "Timestamp": "2024-06-26 11:42:43"}]

    return datasets

def reset():

    """datasets = {"esol": ["smiles", "ESOL predicted log solubility in mols per litre", "data/esol", "2024-06-26 11:36:46.509324"],
           "freesolv": ["smiles", "expt", "data/freesolv", "2024-06-26 11:37:37.393273"],
           "lipo": ["smiles", "y", "data/lipo", "2024-06-26 11:37:37.393273"],
           "hiv": ["smiles", "HIV_active", "data/hiv",  "2024-06-26 11:37:37.393273"],
           "bace": ["smiles", "Class", "data/bace", "2024-06-26 11:38:40.058354"],
           "bbbp": ["smiles", "p_np", "data/bbbp","2024-06-26 11:38:40.058354"],
           "clintox": ["smiles", "CT_TOX", "data/clintox","2024-06-26 11:38:40.058354"],
           "sider": ["smiles","1:", "data/sider","2024-06-26 11:38:40.058354"],
           "tox21": ["smiles",":-2", "data/tox21","2024-06-26 11:38:40.058354"]
           }"""

    datasets = [
      {"Dataset": "hiv", "Input": "smiles", "Output": "HIV_active", "Path": "data/hiv", "Timestamp": "2024-06-26 11:27:37"},
      {"Dataset": "esol", "Input": "smiles", "Output": "ESOL predicted log solubility in mols per litre", "Path": "data/esol", "Timestamp": "2024-06-26 11:31:46"},
      {"Dataset": "freesolv", "Input": "smiles", "Output": "expt", "Path": "data/freesolv", "Timestamp": "2024-06-26 11:33:47"},
      {"Dataset": "lipo", "Input": "smiles", "Output": "y", "Path": "data/lipo", "Timestamp": "2024-06-26 11:34:37"},
      {"Dataset": "bace", "Input": "smiles", "Output": "Class", "Path": "data/bace", "Timestamp": "2024-06-26 11:36:40"},
      {"Dataset": "bbbp", "Input": "smiles", "Output": "p_np", "Path": "data/bbbp", "Timestamp": "2024-06-26 11:39:23"},
      {"Dataset": "clintox", "Input": "smiles", "Output": "CT_TOX", "Path": "data/clintox", "Timestamp": "2024-06-26 11:42:43"},
      #{"Dataset": "sider", "Input": "smiles", "Output": "1:", "path": "data/sider", "Timestamp": "2024-06-26 11:38:40.058354"},
      #{"Dataset": "tox21", "Input": "smiles", "Output": ":-2", "path": "data/tox21", "Timestamp": "2024-06-26 11:38:40.058354"}
    ]

    models = [{"Name": "bart", "Description": "BART model for string based SELFIES modality",
      "Timestamp": "2024-06-21 12:32:20"},
     {"Name": "mol-xl", "Description": "MolFormer model for string based SMILES modality",
      "Timestamp": "2024-06-21 12:35:56"},
     {"Name": "mhg", "Description": "MHG", "Timestamp": "2024-07-10 00:09:42"},
     {"Name": "spec-gru", "Description": "Spectrum modality with GRU", "Timestamp": "2024-07-10 00:09:42"},
     {"Name": "spec-lstm", "Description": "Spectrum modality with LSTM", "Timestamp": "2024-07-10 00:09:54"},
     {"Name": "3d-vae", "Description": "VAE model for 3D atom positions", "Timestamp": "2024-07-10 00:10:08"}]


    downstream_models = [
        {"Name": "XGBClassifier", "Description": "XG Boost Classifier",
         "Timestamp": "2024-06-21 12:31:20"},
        {"Name": "XGBRegressor", "Description": "XG Boost Regressor",
         "Timestamp": "2024-06-21 12:32:56"},
        {"Name": "2-FNN", "Description": "A two layer feedforward network",
         "Timestamp": "2024-06-24 14:34:16"},
        {"Name": "3-FNN", "Description": "A three layer feedforward network",
         "Timestamp": "2024-06-24 14:38:37"},
    ]

    with open("datasets.json", "w") as outfile:
        json.dump(datasets, outfile)

    with open("models.json", "w") as outfile:
        json.dump(models, outfile)

    with open("downstream_models.json", "w") as outfile:
        json.dump(downstream_models, outfile)

def update_data_list(list_data):
    #datasets[list_data[0]] = list_data[1:]

    with open("datasets.json", "w") as outfile:
        json.dump(datasets, outfile)

    avail_models_data()

def update_model_list(list_model):
    #models[list_model[0]] = list_model[1]

    with open("models.json", "w") as outfile:
        json.dump(list_model, outfile)

    avail_models_data()

def update_downstream_model_list(list_model):
    #models[list_model[0]] = list_model[1]

    with open("downstream_models.json", "w") as outfile:
        json.dump(list_model, outfile)

    avail_models_data()

avail_models_data()



def get_representation(train_data,test_data,model_type, return_tensor=True):
    alias = {"MHG-GED": "mhg", "SELFIES-TED": "bart", "MolFormer": "mol-xl", "Molformer": "mol-xl", "SMI-TED": "smi-ted"}
    if model_type in alias.keys():
        model_type = alias[model_type]

    if model_type == "mhg":
        model = mhg.load("../models/mhg_model/pickles/mhggnn_pretrained_model_0724_2023.pickle")
        with torch.no_grad():
            train_emb = model.encode(train_data)
            x_batch = torch.stack(train_emb)

            test_emb = model.encode(test_data)
            x_batch_test = torch.stack(test_emb)
        if not return_tensor:
            x_batch = pd.DataFrame(x_batch)
            x_batch_test = pd.DataFrame(x_batch_test)


    elif model_type == "bart":
        model = bart()
        model.load()
        x_batch = model.encode(train_data, return_tensor=return_tensor)
        x_batch_test = model.encode(test_data, return_tensor=return_tensor)

    elif model_type == "smi-ted":
        model = load_smi_ted(folder='../models/smi_ted/smi_ted_light', ckpt_filename='smi-ted-Light_40.pt')
        with torch.no_grad():
            x_batch = model.encode(train_data, return_torch=return_tensor)
            x_batch_test = model.encode(test_data, return_torch=return_tensor)

    elif model_type == "mol-xl":
        model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                          trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        if type(train_data) == list:
            inputs = tokenizer(train_data, padding=True, return_tensors="pt")
        else:
            inputs = tokenizer(list(train_data.values), padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        x_batch = outputs.pooler_output

        if type(test_data) == list:
            inputs = tokenizer(test_data, padding=True, return_tensors="pt")
        else:
            inputs = tokenizer(list(test_data.values), padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        x_batch_test = outputs.pooler_output

        if not return_tensor:
            x_batch = pd.DataFrame(x_batch)
            x_batch_test = pd.DataFrame(x_batch_test)
    
    elif model_type == 'Mordred':
        all_data = train_data + test_data
        calc = Calculator(descriptors, ignore_3D=True)
        mol_list = [Chem.MolFromSmiles(sm) for sm in all_data]
        x_all = calc.pandas(mol_list)
        print (f'original mordred fv dim: {x_all.shape}')
        
        for j in x_all.columns:
            for k in range(len(x_all[j])):
                i = x_all.loc[k, j]
                if type(i) is mordred.error.Missing or type(i) is mordred.error.Error:
                    x_all.loc[k, j] = np.nan
                    
        x_all.dropna(how="any", axis = 1, inplace=True)
        print (f'Nan excluded mordred fv dim: {x_all.shape}')
        
        x_batch = x_all.iloc[:len(train_data)]
        x_batch_test = x_all.iloc[len(train_data):]
        # print(f'x_batch: {len(x_batch)}, x_batch_test: {len(x_batch_test)}')
        
    elif model_type == 'MorganFingerprint':
        params = {'radius':2, 'nBits':1024}
        
        mol_train = [Chem.MolFromSmiles(sm) for sm in train_data]
        mol_test = [Chem.MolFromSmiles(sm) for sm in test_data]
        
        x_batch = []
        for mol in mol_train:
            info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, **params, bitInfo=info)
            vector = list(fp)
            x_batch.append(vector)
        x_batch = pd.DataFrame(x_batch)
        
        x_batch_test = []
        for mol in mol_test:
            info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, **params, bitInfo=info)
            vector = list(fp)
            x_batch_test.append(vector)
        x_batch_test = pd.DataFrame(x_batch_test)

    return x_batch, x_batch_test

def single_modal(model,dataset=None, downstream_model=None,params=None, x_train=None, x_test=None, y_train=None, y_test=None):
    print(model)
    alias = {"MHG-GED":"mhg", "SELFIES-TED": "bart", "MolFormer":"mol-xl", "Molformer": "mol-xl", "SMI-TED": "smi-ted"}
    data = avail_models(raw=True)
    df = pd.DataFrame(data)
    #print(list(df["Name"].values))
    
    if model in list(df["Name"].values):
        model_type = model
    elif alias[model] in list(df["Name"].values):
            model_type = alias[model]
    else:
        print("Model not available")
        return
    

    data = avail_datasets()
    df = pd.DataFrame(data)
    #print(list(df["Dataset"].values))

    if dataset in list(df["Dataset"].values):
        task = dataset
        with open(f"representation/{task}_{model_type}.pkl", "rb") as f1:
            x_batch, y_batch, x_batch_test, y_batch_test = pickle.load(f1)
        print(f" Representation loaded successfully")

    elif x_train==None:

        print("Custom Dataset")
        #return
        components = dataset.split(",")
        train_data = pd.read_csv(components[0])[components[2]]
        test_data = pd.read_csv(components[1])[components[2]]

        y_batch = pd.read_csv(components[0])[components[3]]
        y_batch_test = pd.read_csv(components[1])[components[3]]


        x_batch,  x_batch_test = get_representation(train_data,test_data,model_type)



        print(f" Representation loaded successfully")

    else:

        y_batch = y_train
        y_batch_test = y_test
        x_batch, x_batch_test = get_representation(x_train, x_test, model_type)
    
    # exclude row containing Nan value
    if isinstance(x_batch, torch.Tensor):
        x_batch = pd.DataFrame(x_batch)    
    nan_indices = x_batch.index[x_batch.isna().any(axis=1)]
    if len(nan_indices) > 0:
        x_batch.dropna(inplace = True)
        for index in sorted(nan_indices, reverse=True):
            del y_batch[index]
        print(f'x_batch Nan index: {nan_indices}')
        print(f'x_batch shape: {x_batch.shape}, y_batch len: {len(y_batch)}')
            
    if isinstance(x_batch_test, torch.Tensor):
        x_batch_test = pd.DataFrame(x_batch_test)
    nan_indices = x_batch_test.index[x_batch_test.isna().any(axis=1)]
    if len(nan_indices) > 0:
        x_batch_test.dropna(inplace = True)
        for index in sorted(nan_indices, reverse=True):
            del y_batch_test[index]
        print(f'x_batch_test Nan index: {nan_indices}')
        print(f'x_batch_test shape: {x_batch_test.shape}, y_batch_test len: {len(y_batch_test)}')

    print(f" Calculating ROC AUC Score ...")

    if downstream_model == "XGBClassifier":
        if params == None:
            xgb_predict_concat = XGBClassifier()
        else:
            xgb_predict_concat = XGBClassifier(**params) # n_estimators=5000, learning_rate=0.01, max_depth=10
        xgb_predict_concat.fit(x_batch, y_batch)

        y_prob = xgb_predict_concat.predict_proba(x_batch_test)[:, 1]

        roc_auc = roc_auc_score(y_batch_test, y_prob)
        fpr, tpr, _ = roc_curve(y_batch_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        try:
            with open(f"plot_emb/{task}_{model_type}.pkl", "rb") as f1:
                class_0,class_1 = pickle.load(f1)
        except:
            print("Generating latent plots")
            reducer = umap.UMAP(metric='euclidean', n_neighbors=10, n_components=2, low_memory=True, min_dist=0.1,
                                verbose=False)
            n_samples = np.minimum(1000, len(x_batch))

            try:x = y_batch.values[:n_samples]
            except: x = y_batch[:n_samples]
            index_0 = [index for index in range(len(x)) if x[index] == 0]
            index_1 = [index for index in range(len(x)) if x[index] == 1]

            try:
                features_umap = reducer.fit_transform(x_batch[:n_samples])
                class_0 = features_umap[index_0]
                class_1 = features_umap[index_1]
            except:
                class_0 = []
                class_1 = []
            print("Generating latent plots : Done")

        #vizualize(roc_auc,fpr, tpr, x_batch, y_batch )

        result = f"ROC-AUC Score: {roc_auc:.4f}"

        return result, roc_auc,fpr, tpr, class_0, class_1

    elif downstream_model == "DefaultClassifier":
        xgb_predict_concat = XGBClassifier() # n_estimators=5000, learning_rate=0.01, max_depth=10
        xgb_predict_concat.fit(x_batch, y_batch)

        y_prob = xgb_predict_concat.predict_proba(x_batch_test)[:, 1]

        roc_auc = roc_auc_score(y_batch_test, y_prob)
        fpr, tpr, _ = roc_curve(y_batch_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        try:
            with open(f"plot_emb/{task}_{model_type}.pkl", "rb") as f1:
                class_0,class_1 = pickle.load(f1)
        except:
            print("Generating latent plots")
            reducer = umap.UMAP(metric='euclidean', n_neighbors=  10, n_components=2, low_memory=True, min_dist=0.1, verbose=False)
            n_samples = np.minimum(1000,len(x_batch))

            try:
                x = y_batch.values[:n_samples]
            except:
                x = y_batch[:n_samples]

            try:
                features_umap = reducer.fit_transform(x_batch[:n_samples])
                index_0 = [index for index in range(len(x)) if x[index] == 0]
                index_1 = [index for index in range(len(x)) if x[index] == 1]

                class_0 = features_umap[index_0]
                class_1 = features_umap[index_1]
            except:
                class_0 = []
                class_1 = []

            print("Generating latent plots : Done")

        #vizualize(roc_auc,fpr, tpr, x_batch, y_batch )

        result = f"ROC-AUC Score: {roc_auc:.4f}"

        return result, roc_auc,fpr, tpr, class_0, class_1
    
    elif downstream_model == "SVR":
        if params == None:
            regressor = SVR()
        else:            
            regressor = SVR(**params)
        model = TransformedTargetRegressor(regressor= regressor,
                                                transformer = MinMaxScaler(feature_range=(-1, 1))
                                                ).fit(x_batch,y_batch)
        
        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))
        
        print(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        print("Generating latent plots")
        reducer = umap.UMAP(metric='euclidean', n_neighbors=10, n_components=2, low_memory=True, min_dist=0.1,
                            verbose=False)
        n_samples = np.minimum(1000, len(x_batch))

        try: x = y_batch.values[:n_samples]
        except: x = y_batch[:n_samples]
        #index_0 = [index for index in range(len(x)) if x[index] == 0]
        #index_1 = [index for index in range(len(x)) if x[index] == 1]

        try:
            features_umap = reducer.fit_transform(x_batch[:n_samples])
            class_0 = features_umap#[index_0]
            class_1 = features_umap#[index_1]
        except:
            class_0 = []
            class_1 = []
        print("Generating latent plots : Done")
        
        return result, RMSE_score,y_batch_test, y_prob, class_0, class_1

    elif downstream_model == "Kernel Ridge":
        if params == None:
            regressor = KernelRidge()
        else:
            regressor = KernelRidge(**params)
        model = TransformedTargetRegressor(regressor=regressor,
                                           transformer=MinMaxScaler(feature_range=(-1, 1))
                                           ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        print(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        print("Generating latent plots")
        reducer = umap.UMAP(metric='euclidean', n_neighbors=10, n_components=2, low_memory=True, min_dist=0.1,
                            verbose=False)
        n_samples = np.minimum(1000, len(x_batch))
        features_umap = reducer.fit_transform(x_batch[:n_samples])
        try: x = y_batch.values[:n_samples]
        except: x = y_batch[:n_samples]
        # index_0 = [index for index in range(len(x)) if x[index] == 0]
        # index_1 = [index for index in range(len(x)) if x[index] == 1]

        class_0 = features_umap#[index_0]
        class_1 = features_umap#[index_1]
        print("Generating latent plots : Done")

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1


    elif downstream_model == "Linear Regression":
        if params == None:
            regressor = LinearRegression()
        else:
            regressor = LinearRegression(**params)
        model = TransformedTargetRegressor(regressor=regressor,
                                           transformer=MinMaxScaler(feature_range=(-1, 1))
                                           ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        print(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        print("Generating latent plots")
        reducer = umap.UMAP(metric='euclidean', n_neighbors=10, n_components=2, low_memory=True, min_dist=0.1,
                            verbose=False)
        n_samples = np.minimum(1000, len(x_batch))
        features_umap = reducer.fit_transform(x_batch[:n_samples])
        try:x = y_batch.values[:n_samples]
        except: x = y_batch[:n_samples]
        # index_0 = [index for index in range(len(x)) if x[index] == 0]
        # index_1 = [index for index in range(len(x)) if x[index] == 1]

        class_0 = features_umap#[index_0]
        class_1 = features_umap#[index_1]
        print("Generating latent plots : Done")

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1


    elif downstream_model == "DefaultRegressor":
        regressor = SVR(kernel="rbf", degree=3, C=5, gamma="scale", epsilon=0.01)
        model = TransformedTargetRegressor(regressor=regressor,
                                           transformer=MinMaxScaler(feature_range=(-1, 1))
                                           ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        print(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        print("Generating latent plots")
        reducer = umap.UMAP(metric='euclidean', n_neighbors=10, n_components=2, low_memory=True, min_dist=0.1,
                            verbose=False)
        n_samples = np.minimum(1000, len(x_batch))
        features_umap = reducer.fit_transform(x_batch[:n_samples])
        try:x = y_batch.values[:n_samples]
        except: x = y_batch[:n_samples]
        # index_0 = [index for index in range(len(x)) if x[index] == 0]
        # index_1 = [index for index in range(len(x)) if x[index] == 1]

        class_0 = features_umap#[index_0]
        class_1 = features_umap#[index_1]
        print("Generating latent plots : Done")

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1
        

def multi_modal(model_list,dataset=None, downstream_model=None,params=None, x_train=None, x_test=None, y_train=None, y_test=None):
    #print(model_list)
    data = avail_datasets()
    df = pd.DataFrame(data)
    list(df["Dataset"].values)

    if dataset in list(df["Dataset"].values):
        task = dataset
        predefined = True
    elif x_train==None:
        predefined = False
        components = dataset.split(",")
        train_data = pd.read_csv(components[0])[components[2]]
        test_data = pd.read_csv(components[1])[components[2]]

        y_batch = pd.read_csv(components[0])[components[3]]
        y_batch_test = pd.read_csv(components[1])[components[3]]

        print("Custom Dataset loaded")
    else:
        predefined = False
        y_batch = y_train
        y_batch_test = y_test
        train_data = x_train
        test_data = x_test

    data = avail_models(raw=True)
    df = pd.DataFrame(data)
    list(df["Name"].values)

    alias = {"MHG-GED":"mhg", "SELFIES-TED": "bart", "MolFormer":"mol-xl",  "Molformer": "mol-xl","SMI-TED":"smi-ted", "Mordred": "Mordred", "MorganFingerprint": "MorganFingerprint"}
    #if set(model_list).issubset(list(df["Name"].values)):
    if set(model_list).issubset(list(alias.keys())):
        for i, model in enumerate(model_list):
            if model in alias.keys():
                model_type = alias[model]
            else:
                model_type = model

            if i == 0:
                if predefined:
                    with open(f"representation/{task}_{model_type}.pkl", "rb") as f1:
                        x_batch, y_batch, x_batch_test, y_batch_test = pickle.load(f1)
                    print(f" Loaded representation/{task}_{model_type}.pkl")
                else:
                    x_batch, x_batch_test = get_representation(train_data, test_data, model_type)
                    x_batch = pd.DataFrame(x_batch)
                    x_batch_test = pd.DataFrame(x_batch_test)

            else:
                if predefined:
                    with open(f"representation/{task}_{model_type}.pkl", "rb") as f1:
                        x_batch_1, y_batch_1, x_batch_test_1, y_batch_test_1 = pickle.load(f1)
                        print(f" Loaded representation/{task}_{model_type}.pkl")
                else:
                    x_batch_1, x_batch_test_1 = get_representation(train_data, test_data, model_type)
                    x_batch_1 = pd.DataFrame(x_batch_1)
                    x_batch_test_1 = pd.DataFrame(x_batch_test_1)

                x_batch = pd.concat([x_batch, x_batch_1], axis=1)
                x_batch_test = pd.concat([x_batch_test, x_batch_test_1], axis=1)

    else:
        print("Model not available")
        return

    num_columns = x_batch_test.shape[1]
    x_batch_test.columns = [f'{i + 1}' for i in range(num_columns)]

    num_columns = x_batch.shape[1]
    x_batch.columns = [f'{i + 1}' for i in range(num_columns)]
    
    # exclude row containing Nan value
    if isinstance(x_batch, torch.Tensor):
        x_batch = pd.DataFrame(x_batch)    
    nan_indices = x_batch.index[x_batch.isna().any(axis=1)]
    if len(nan_indices) > 0:
        x_batch.dropna(inplace = True)
        for index in sorted(nan_indices, reverse=True):
            del y_batch[index]
        print(f'x_batch Nan index: {nan_indices}')
        print(f'x_batch shape: {x_batch.shape}, y_batch len: {len(y_batch)}')
            
    if isinstance(x_batch_test, torch.Tensor):
        x_batch_test = pd.DataFrame(x_batch_test)
    nan_indices = x_batch_test.index[x_batch_test.isna().any(axis=1)]
    if len(nan_indices) > 0:
        x_batch_test.dropna(inplace = True)
        for index in sorted(nan_indices, reverse=True):
            del y_batch_test[index]
        print(f'x_batch_test Nan index: {nan_indices}')
        print(f'x_batch_test shape: {x_batch_test.shape}, y_batch_test len: {len(y_batch_test)}')

    print(f"Representations loaded successfully")
    try:
        with open(f"plot_emb/{task}_multi.pkl", "rb") as f1:
            class_0, class_1 = pickle.load(f1)
    except:
        print("Generating latent plots")
        reducer = umap.UMAP(metric='euclidean', n_neighbors=10, n_components=2, low_memory=True, min_dist=0.1,
                            verbose=False)
        n_samples = np.minimum(1000, len(x_batch))
        features_umap = reducer.fit_transform(x_batch[:n_samples])

        if "Classifier" in downstream_model:
            try: x = y_batch.values[:n_samples]
            except: x = y_batch[:n_samples]
            index_0 = [index for index in range(len(x)) if x[index] == 0]
            index_1 = [index for index in range(len(x)) if x[index] == 1]

            class_0 = features_umap[index_0]
            class_1 = features_umap[index_1]

        else:
            class_0 = features_umap
            class_1 = features_umap

        print("Generating latent plots : Done")

    print(f" Calculating ROC AUC Score ...")


    if downstream_model == "XGBClassifier":
        if params == None:
            xgb_predict_concat = XGBClassifier()
        else:            
            xgb_predict_concat = XGBClassifier(**params)#n_estimators=5000, learning_rate=0.01, max_depth=10)
        xgb_predict_concat.fit(x_batch, y_batch)

        y_prob = xgb_predict_concat.predict_proba(x_batch_test)[:, 1]


        roc_auc = roc_auc_score(y_batch_test, y_prob)
        fpr, tpr, _ = roc_curve(y_batch_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        #vizualize(roc_auc,fpr, tpr, x_batch, y_batch )

        #vizualize(x_batch_test, y_batch_test)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        result = f"ROC-AUC Score: {roc_auc:.4f}"

        return result, roc_auc,fpr, tpr, class_0, class_1

    elif downstream_model == "DefaultClassifier":
        xgb_predict_concat = XGBClassifier()#n_estimators=5000, learning_rate=0.01, max_depth=10)
        xgb_predict_concat.fit(x_batch, y_batch)

        y_prob = xgb_predict_concat.predict_proba(x_batch_test)[:, 1]


        roc_auc = roc_auc_score(y_batch_test, y_prob)
        fpr, tpr, _ = roc_curve(y_batch_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        #vizualize(roc_auc,fpr, tpr, x_batch, y_batch )

        #vizualize(x_batch_test, y_batch_test)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        result = f"ROC-AUC Score: {roc_auc:.4f}"

        return result, roc_auc,fpr, tpr, class_0, class_1

    elif downstream_model == "SVR":
        if params == None:
            regressor = SVR()
        else:
            regressor = SVR(**params)
        model = TransformedTargetRegressor(regressor= regressor,
                                                transformer = MinMaxScaler(feature_range=(-1, 1))
                                                ).fit(x_batch,y_batch)
        
        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))
        
        print(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"
        
        return result, RMSE_score,y_batch_test, y_prob, class_0, class_1

    elif downstream_model == "Linear Regression":
        if params == None:
            regressor = LinearRegression()
        else:
            regressor = LinearRegression(**params)
        model = TransformedTargetRegressor(regressor=regressor,
                                           transformer=MinMaxScaler(feature_range=(-1, 1))
                                           ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        print(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1

    elif downstream_model == "Kernel Ridge":
        if params == None:
            regressor = KernelRidge()
        else:
            regressor = KernelRidge(**params)
        model = TransformedTargetRegressor(regressor=regressor,
                                           transformer=MinMaxScaler(feature_range=(-1, 1))
                                           ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        print(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1

    elif downstream_model == "DefaultRegressor":
        regressor = SVR(kernel="rbf", degree=3, C=5, gamma="scale", epsilon=0.01)
        model = TransformedTargetRegressor(regressor=regressor,
                                           transformer=MinMaxScaler(feature_range=(-1, 1))
                                           ).fit(x_batch, y_batch)

        y_prob = model.predict(x_batch_test)
        RMSE_score = np.sqrt(mean_squared_error(y_batch_test, y_prob))

        print(f"RMSE Score: {RMSE_score:.4f}")
        result = f"RMSE Score: {RMSE_score:.4f}"

        return result, RMSE_score, y_batch_test, y_prob, class_0, class_1



def finetune_optuna(x_batch,y_batch, x_batch_test, y_test ):
    print(f" Finetuning with Optuna and calculating ROC AUC Score ...")
    X_train = x_batch.values
    y_train = y_batch.values
    X_test = x_batch_test.values
    y_test = y_test.values
    def objective(trial):
        # Define parameters to be optimized
        params = {
            # 'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 1000, 10000),
            # 'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            # 'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
            'max_depth': trial.suggest_int('max_depth', 1, 12),
            # 'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
            # 'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            # 'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        }

        # Train XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(params, dtrain)

        # Predict probabilities
        y_pred = model.predict(dtest)

        # Calculate ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred)
        print("ROC_AUC : ", roc_auc)

        return roc_auc

def add_new_model():
    models = avail_models(raw=True)

    # Function to display models
    def display_models():
        for model in models:
            model_display = f"Name: {model['Name']}, Description: {model['Description']}, Timestamp: {model['Timestamp']}"
            print(model_display)

    # Function to update models
    def update_models(new_name, new_description, new_path):
        new_model = {
            "Name": new_name,
            "Description": new_description,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #"path": new_path
        }
        models.append(new_model)
        with open("models.json", "w") as outfile:
            json.dump(models, outfile)

        print("Model uploaded and updated successfully!")
        list_models()
        #display_models()

    # Widgets
    name_text = widgets.Text(description="Name:", layout=Layout(width='50%'))
    description_text = widgets.Text(description="Description:", layout=Layout(width='50%'))
    path_text = widgets.Text(description="Path:", layout=Layout(width='50%'))

    def browse_callback(b):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(title="Select a Model File")
        if file_path:
            path_text.value = file_path

    browse_button = widgets.Button(description="Browse")
    browse_button.on_click(browse_callback)

    def submit_callback(b):
        update_models(name_text.value, description_text.value, path_text.value)

    submit_button = widgets.Button(description="Submit")
    submit_button.on_click(submit_callback)

    # Display widgets
    display(VBox([name_text, description_text, path_text, browse_button, submit_button]))


def add_new_dataset():
    # Sample data
    datasets = avail_datasets()

    # Function to display models
    def display_datasets():
        for dataset in datasets:
            dataset_display = f"Name: {dataset['Dataset']}, Input: {dataset['Input']},Output: {dataset['Output']},Path: {dataset['Path']}, Timestamp: {dataset['Timestamp']}"

    # Function to update models
    def update_datasets(new_dataset, new_input, new_output, new_path):
        new_model = {
            "Dataset": new_dataset,
            "Input": new_input,
            "Output": new_output,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Path": os.path.basename(new_path)
        }
        datasets.append(new_model)
        with open("datasets.json", "w") as outfile:
            json.dump(datasets, outfile)

        print("Dataset uploaded and updated successfully!")
        list_data()


    # Widgets
    dataset_text = widgets.Text(description="Dataset:", layout=Layout(width='50%'))
    input_text = widgets.Text(description="Input:", layout=Layout(width='50%'))
    output_text = widgets.Text(description="Output:", layout=Layout(width='50%'))
    path_text = widgets.Text(description="Path:", layout=Layout(width='50%'))

    def browse_callback(b):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(title="Select a Dataset File")
        if file_path:
            path_text.value = file_path

    browse_button = widgets.Button(description="Browse")
    browse_button.on_click(browse_callback)

    def submit_callback(b):
        update_datasets(dataset_text.value, input_text.value, output_text.value, path_text.value)

    submit_button = widgets.Button(description="Submit")
    submit_button.on_click(submit_callback)

    display(VBox([dataset_text, input_text, output_text, path_text, browse_button, submit_button]))



