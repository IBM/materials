import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import time
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.base import clone

from . import load as tdims

def get_representation(sm_list, model = 'tdims', radius=1, func_dis=-2, func_merge=sum, fragment_set=True, atom_set=True, fingerprint_set=True, display=False):
    if model == "tdims":
        start_time = time.time()
        
        emb, key_all = tdims.encode(sm_list, radius=radius, func_dis=func_dis, func_merge=func_merge, fragment_set=fragment_set, atom_set=atom_set, fingerprint_set=fingerprint_set)
        
        if display:
            print(f'Full embedding shape: {emb.shape}')
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        if display:
            print(f'Execution time for full embedding: {elapsed_time: .6f} sec')
                
        return emb, list(key_all.keys())

    else:
        raise ValueError("Invalid model input")
    

def get_representation_with_fs_selection(sm_list, y, model="tdims", radius=1, func_dis=-2, func_merge=sum, fragment_set=True, atom_set=True, fingerprint_set=True, display=False):
    if model == 'tdims':
        x_all, key_all = get_representation(sm_list, model, radius=radius, func_dis=func_dis, func_merge=func_merge, fragment_set=fragment_set, atom_set=atom_set, fingerprint_set=fingerprint_set)
        
        start_time = time.time()
        selector = SelectFromModel(estimator=Lasso(random_state=0, alpha=0.0001, max_iter=100000, tol=1e-4), threshold="mean").fit(x_all, y)    
        x_slc = selector.transform(x_all)
        key_slc = np.array(key_all)[selector.get_support()]
        
        end_time = time.time()
        
            
        if display:                
            print(f'\nFeature were selected from {x_all.shape} to {x_slc.shape}')
        
        elapsed_time = end_time - start_time
        if display:
            print(f'Execution time for feature selection: {elapsed_time: .6f} sec')
                
        return x_slc, list(key_slc), key_all



def run_tdims_regression_cv(
    sm_list,
    y,
    *,
    radius=1,
    func_dis=-2,
    func_merge=sum,
    fragment_set=True,
    atom_set=True,
    fingerprint_set=True,
    cv=3,
    n_repeats=2,
    inner_cv=3,
    random_state=0,
    model_name="lasso",
    mode="quick",
    use_feature_selection=True,
    fs_alpha=0.0001,
    fs_threshold="mean",
    return_fold_details=True,
    fit_final_estimator=True,
    n_jobs=-1,
    verbose=0,
):
    """
    Run nested CV for TDiMS regression without data leakage.

    Procedure
    ---------
    1. Generate TDiMS descriptors from SMILES only.
    2. Split data by outer RepeatedKFold.
    3. Inside each outer-train fold:
       - fit feature selection using outer-train only
       - perform hyperparameter optimization by inner CV
    4. Evaluate the best estimator on the outer-test fold only.

    Parameters
    ----------
    sm_list : array-like
        List-like object of SMILES strings.
    y : array-like
        Target values.
    radius, func_dis, func_merge, fragment_set, atom_set, fingerprint_set
        Parameters for TDiMS descriptor generation.
    cv : int, default=3
        Number of folds in outer RepeatedKFold.
    n_repeats : int, default=2
        Number of repeats in outer RepeatedKFold.
    inner_cv : int, default=3
        Number of folds in inner KFold for hyperparameter optimization.
    random_state : int, default=0
        Random seed.
    model_name : {"lasso", "ridge", "elasticnet", "rf"}, default="lasso"
        Regression model family to optimize.
    mode : {"full", "quick"}, default="quick"
        Hyperparameter search mode.

        - "full":
          Full search space for the main experiments.
        - "quick":
          Reduced search space for fast testing.

    use_feature_selection : bool, default=True
        If True, apply Lasso-based feature selection inside each outer fold.
    fs_alpha : float, default=0.0001
        Alpha for Lasso used in feature selection.
    fs_threshold : str or float, default="mean"
        Threshold for SelectFromModel.
    return_fold_details : bool, default=True
        If True, return detailed information for each outer fold.
    fit_final_estimator : bool, default=True
        If True, refit the optimized model on all data after CV.
    n_jobs : int, default=-1
        Number of parallel jobs for GridSearchCV.
    verbose : int, default=0
        Verbosity for GridSearchCV.

    Returns
    -------
    result_dict : dict
        Dictionary containing:
        - r2_scores
        - r2_mean
        - r2_std
        - estimators
        - selectors
        - best_params_per_fold
        - final_estimator
        - final_selector
        - final_best_params
        - fold_details (optional)
    """
    sm_list = np.asarray(sm_list)
    y = np.asarray(y)

    if sm_list.shape[0] != y.shape[0]:
        raise ValueError("sm_list and y must have the same length.")

    mode = str(mode).lower().strip()
    if mode not in {"full", "quick"}:
        raise ValueError("mode must be either 'full' or 'quick'.")

    # ---------------------------------------------------------
    # Descriptor generation from SMILES only
    # ---------------------------------------------------------
    X_all, key_all = get_representation(
        sm_list,
        model="tdims",
        radius=radius,
        func_dis=func_dis,
        func_merge=func_merge,
        fragment_set=fragment_set,
        atom_set=atom_set,
        fingerprint_set=fingerprint_set,
    )
    X_all = np.asarray(X_all)
    key_all = np.asarray(key_all)

    print(f"\nFull descriptor matrix shape: {X_all.shape}")

    # ---------------------------------------------------------
    # Hyperparameter search spaces
    # ---------------------------------------------------------
    alpha_lasso_full = np.logspace(-6, 1, 8)
    alpha_en_full = np.logspace(-2, 2, 5)
    alpha_ridge_full = np.logspace(-2, 3, 6)

    alpha_lasso_quick = np.array([1e-4, 1e-2, 1.0])
    alpha_en_quick = np.array([1e-2, 1e-1, 1.0])
    alpha_ridge_quick = np.array([1e-2, 1e0, 1e2])

    if mode == "quick":
        alpha_lasso = alpha_lasso_quick
        alpha_en = alpha_en_quick
        alpha_ridge = alpha_ridge_quick
        rf_min_samples_split = [2]
        en_l1_ratio = [0.2, 0.8]
    else:
        alpha_lasso = alpha_lasso_full
        alpha_en = alpha_en_full
        alpha_ridge = alpha_ridge_full
        rf_min_samples_split = [2, 3]
        en_l1_ratio = [0.2, 0.4, 0.6, 0.8]

    model_name = str(model_name).lower()

    if model_name == "lasso":
        base_model = Lasso(
            random_state=random_state,
            max_iter=100000,
            tol=1e-4,
        )
        param_grid = {"alpha": alpha_lasso}

    elif model_name == "ridge":
        base_model = Ridge(
            random_state=random_state,
        )
        param_grid = {"alpha": alpha_ridge}

    elif model_name == "elasticnet":
        base_model = ElasticNet(
            random_state=random_state,
            max_iter=100000,
            tol=1e-4,
        )
        param_grid = {
            "alpha": alpha_en,
            "l1_ratio": en_l1_ratio,
        }

    elif model_name in {"rf", "randomforest", "random_forest"}:
        base_model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
        )
        param_grid = {
            "min_samples_split": rf_min_samples_split,
        }

    else:
        raise ValueError(
            "model_name must be one of: "
            "'lasso', 'ridge', 'elasticnet', 'rf'"
        )

    # ---------------------------------------------------------
    # Outer CV
    # ---------------------------------------------------------
    outer_cv = RepeatedKFold(
        n_splits=cv,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_r2_scores = []
    fold_details = []
    estimators = []
    selectors = []
    best_params_per_fold = []

    start_time = time.time()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, y)):
            X_train = X_all[train_idx]
            X_test = X_all[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            selected_keys = key_all.copy()
            selector_fitted = None

            # -------------------------------------------------
            # Feature selection on outer-train only
            # -------------------------------------------------
            if use_feature_selection:
                selector_fitted = SelectFromModel(
                    estimator=Lasso(
                        random_state=0,
                        alpha=fs_alpha,
                        max_iter=100000,
                        tol=1e-4,
                    ),
                    threshold=fs_threshold,
                )
                selector_fitted.fit(X_train, y_train)

                X_train_used = selector_fitted.transform(X_train)
                X_test_used = selector_fitted.transform(X_test)
                selected_keys = key_all[selector_fitted.get_support()]
            else:
                X_train_used = X_train
                X_test_used = X_test

            # -------------------------------------------------
            # Inner CV for hyperparameter optimization
            # -------------------------------------------------
            inner_splitter = KFold(
                n_splits=inner_cv,
                shuffle=True,
                random_state=random_state,
            )

            gscv = GridSearchCV(
                estimator=clone(base_model),
                param_grid=param_grid,
                scoring="r2",
                cv=inner_splitter,
                n_jobs=n_jobs,
                refit=True,
                verbose=verbose,
            )
            gscv.fit(X_train_used, y_train)

            best_estimator = gscv.best_estimator_
            best_params = dict(gscv.best_params_)

            # -------------------------------------------------
            # Evaluate on outer-test only
            # -------------------------------------------------
            fold_r2 = best_estimator.score(X_test_used, y_test)
            fold_r2_scores.append(fold_r2)

            estimators.append(best_estimator)
            selectors.append(selector_fitted)
            best_params_per_fold.append(best_params)

            if return_fold_details:
                fold_details.append(
                    {
                        "fold": fold_idx,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "r2": float(fold_r2),
                        "n_features_before_fs": int(X_train.shape[1]),
                        "n_features_after_fs": int(X_train_used.shape[1]),
                        "selected_keys": list(selected_keys),
                        "best_params": best_params,
                        "best_inner_cv_score": float(gscv.best_score_),
                    }
                )

        elapsed_time = time.time() - start_time
        fold_r2_scores = np.asarray(fold_r2_scores, dtype=float)

        # -----------------------------------------------------
        # Final estimator fitted on all data
        # -----------------------------------------------------
        final_selector = None
        final_estimator = None
        final_selected_keys = list(key_all)
        final_best_params = None
        final_X_selected = X_all

        if fit_final_estimator:
            if use_feature_selection:
                final_selector = SelectFromModel(
                    estimator=Lasso(
                        random_state=0,
                        alpha=fs_alpha,
                        max_iter=100000,
                        tol=1e-4,
                    ),
                    threshold=fs_threshold,
                )
                final_selector.fit(X_all, y)
                X_all_used = final_selector.transform(X_all)
                final_selected_keys = list(key_all[final_selector.get_support()])
                final_X_selected = X_all_used
            else:
                X_all_used = X_all
                final_X_selected = X_all

            final_inner_splitter = KFold(
                n_splits=inner_cv,
                shuffle=True,
                random_state=random_state,
            )

            final_gscv = GridSearchCV(
                estimator=clone(base_model),
                param_grid=param_grid,
                scoring="r2",
                cv=final_inner_splitter,
                n_jobs=n_jobs,
                refit=True,
                verbose=verbose,
            )
            final_gscv.fit(X_all_used, y)

            final_estimator = final_gscv.best_estimator_
            final_best_params = dict(final_gscv.best_params_)

    print("\n===== TDiMS regression CV result =====")
    print(f"Model family        : {model_name}")
    print(f"Mode                : {mode}")
    print(f"Descriptor shape    : {X_all.shape}")
    print(f"Use FS              : {use_feature_selection}")
    print(f"Outer CV            : RepeatedKFold(n_splits={cv}, n_repeats={n_repeats})")
    print(f"Inner CV            : KFold(n_splits={inner_cv}, shuffle=True)")
    print(f"R2 scores           : {np.round(fold_r2_scores, 4)}")
    print(f"Mean R2 +/- std     : {np.mean(fold_r2_scores):.4f} +/- {np.std(fold_r2_scores, ddof=1):.4f}")
    print(f"Execution time      : {elapsed_time:.6f} sec")

    result_dict = {
        "r2_scores": fold_r2_scores,
        "r2_mean": float(np.mean(fold_r2_scores)),
        "r2_std": float(np.std(fold_r2_scores, ddof=1)),
        "X_shape": X_all.shape,
        "model_name": model_name,
        "mode": mode,
        "cv": cv,
        "n_repeats": n_repeats,
        "inner_cv": inner_cv,
        "use_feature_selection": use_feature_selection,
        "radius": radius,
        "func_dis": func_dis,
        "func_merge": getattr(func_merge, "__name__", str(func_merge)),
        "fragment_set": fragment_set,
        "elapsed_time_sec": float(elapsed_time),
        "estimators": estimators,
        "selectors": selectors,
        "best_params_per_fold": best_params_per_fold,
        "feature_names_all": list(key_all),
        "final_estimator": final_estimator,
        "final_selector": final_selector,
        "final_selected_keys": final_selected_keys,
        "final_best_params": final_best_params,
        "final_X_selected": final_X_selected,
        "final_y": np.asarray(y),
        "search_space": {
            "alpha_lasso": alpha_lasso.tolist(),
            "alpha_en": alpha_en.tolist(),
            "alpha_ridge": alpha_ridge.tolist(),
            "rf_min_samples_split": rf_min_samples_split,
            "en_l1_ratio": en_l1_ratio,
        },
    }

    if return_fold_details:
        result_dict["fold_details"] = fold_details

    return result_dict