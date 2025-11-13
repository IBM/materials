from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import load as tdims

def get_representation(sm_list, model = 'tdims', radius=1, func_dis=-2, func_merge=sum, fragment_set=True, atom_set=True, fingerprint_set=True):
    if model == "tdims":
        start_time = time.time()
        
        emb, key_all = tdims.encode(sm_list, radius=radius, func_dis=func_dis, func_merge=func_merge, fragment_set=fragment_set, atom_set=atom_set, fingerprint_set=fingerprint_set)
        scaler = StandardScaler() 
        scaler.fit(emb)
        emb = scaler.transform(emb)
        
        print(f'Full embedding shape: {emb.shape}')
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Execution time for full embedding: {elapsed_time: .6f} sec')
                
        return emb, list(key_all.keys())

    else:
        raise ValueError("Invalid model input")
    

def get_representation_with_fs_selection(sm_list, y, reg_model, model="tdims", radius=1, func_dis=-2, func_merge=sum, fragment_set=True, atom_set=True, fingerprint_set=True):
    if model == 'tdims':
        x_all, key_all = get_representation(sm_list, model, radius=radius, func_dis=func_dis, func_merge=func_merge, fragment_set=fragment_set, atom_set=atom_set, fingerprint_set=fingerprint_set)
        
        start_time = time.time()
        kf = RepeatedKFold(n_splits=3, n_repeats=10)
        
        if reg_model == 'RondomForest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            rf = RandomForestRegressor()
            
            grid_search = GridSearchCV(rf, param_grid, cv=5)
            grid_search.fit(x_all, y)
            
            optimized_params = grid_search.best_params_
            rf.set_params(**optimized_params)
            
            selector = SelectFromModel(rf, threshold=None).fit(x_all, y)
            
        else:
            if reg_model == 'Lasso':
                param_grid = {'alpha': np.logspace(-6, 1, 8)}
                search = LassoCV(alphas = np.logspace(-6, 1, 8), cv = kf)
                
            elif reg_model == 'Ridge':
                param_grid = {'alpha':np.logspace(-4, 3, 8)}
                search = RidgeCV(alphas=param_grid['alpha'], cv=kf)
            
            elif reg_model == 'ElasticNet':
                param_grid = {'alpha':np.logspace(-6, 2, 9), 'l1_ratio':np.linspace(0.0, 1.0, 6)}
                search = ElasticNetCV(l1_ratio=param_grid['l1_ratio'], alphas=param_grid['alpha'], cv=kf)
                if isinstance(search.l1_ratio_, np.ndarray):
                    search.l1_ratio_ = search.l1_ratio_[0]
        
            search.fit(x_all, y)
            optimized_params = {'alpha': search.alpha_}
            if reg_model == 'ElasticNet':
                optimized_params['l1_ratio'] = search.l1_ratio_
            
            selector = SelectFromModel(search, threshold=None).fit(x_all, y)
            
        x_slc = selector.transform(x_all)
        key_slc = np.array(key_all)[selector.get_support()]
        
        end_time = time.time()
                
        print(f'\nFeature were selected from {x_all.shape} to {x_slc.shape}')
        print(f'Optimized parameter for {reg_model}: {optimized_params}')
        
        elapsed_time = end_time - start_time
        print(f'Execution time for feature selection: {elapsed_time: .6f} sec')
                
        return x_slc, list(key_slc), key_all, optimized_params
    
    
