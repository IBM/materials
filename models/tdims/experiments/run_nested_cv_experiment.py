#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import os
import re
import tempfile
from itertools import product

import numpy as np
import pandas as pd

from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tdims.sparse_transformers import (
    ClipGreaterThanOneToZero,
    SafeSelectFromModel,
    SparseFeatureFilterVCSR,
    ToCSR,
)
from tdims import tdims_ext

def make_pipeline_template(memory_obj: Optional[Memory] = None) -> Pipeline:
    """
    Build the common scikit-learn pipeline used in the experiments.

    Steps
    -----
    descriptor
        Descriptor generator (TDiMS or benchmark embedding loader).
    to_csr / clip_gt1_to0 / sparse_filter
        Sparse-preprocessing utilities used for TDiMS-style descriptors.
    scaler
        Scaling strategy selected depending on the descriptor family.
    select
        Embedded feature selection step.
    model
        Downstream regression model.
    """
    return Pipeline(
        [
            ("descriptor", None),
            ("to_csr", ToCSR()),
            ("clip_gt1_to0", ClipGreaterThanOneToZero(threshold=1.0)),
            ("sparse_filter", SparseFeatureFilterVCSR(min_samples=1)),
            ("scaler", "passthrough"),
            ("select", SafeSelectFromModel(estimator=Lasso(random_state=0), threshold=None)),
            ("model", Lasso(random_state=0)),
        ],
        memory=memory_obj,
    )


class PandasFunctionTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper around sklearn's FunctionTransformer with two safeguards:

    1. The descriptor function must return a pandas.DataFrame.
    2. Feature columns are aligned between fit() and transform().

    This helps keep descriptor generation stable across CV folds.
    """

    def __init__(self, func: Any = None, kw_args: Any = None):
        self.func = func
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.transformer_ = FunctionTransformer(self.func, kw_args=self.kw_args)
        Xt = self.transformer_.fit_transform(X)
        if not isinstance(Xt, pd.DataFrame):
            raise TypeError("Descriptor function must return a pandas.DataFrame.")
        self.feature_names_out_ = list(Xt.columns)
        return self

    def transform(self, X):
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("fit() must be called before transform().")
        Xt = self.transformer_.transform(X)
        if not isinstance(Xt, pd.DataFrame):
            raise TypeError("Descriptor function must return a pandas.DataFrame.")
        Xtt = Xt.reindex(columns=self.feature_names_out_, fill_value=0)
        return Xtt.values


def _resolve(X: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve the lightweight index dataframe into the original source dataframe.

    Expected input format
    ---------------------
    X.iloc[:, 0]
        CSV path
    X.iloc[:, 1]
        Row indices to read from that CSV
    """
    (path,) = X.iloc[:, 0].unique()
    indices = X.iloc[:, 1].values
    return pd.read_csv(path).iloc[indices]


def tdims_descriptor(
    X: pd.DataFrame,
    radius: int,
    func_dis: int,
    func_merge: Any,
    fragment_set: bool,
) -> pd.DataFrame:
    """
    Generate TDiMS descriptors for the rows specified by X.

    Notes
    -----
    - X is a lightweight dataframe containing the source CSV path and row indices.
    - The returned object must be a pandas.DataFrame for compatibility with
      PandasFunctionTransformer.
    """
    df_raw = _resolve(X)

    tmp_path = None
    try:
        # Kept for compatibility with the original workflow.
        # Remove if no file-based intermediate handling is needed in your setup.
        with tempfile.NamedTemporaryFile(mode="wt", suffix=".csv", delete=False) as fp:
            tmp_path = fp.name
            df_raw.to_csv(fp)

        print(
            f"[INFO] TDiMS | radius={radius}, fragment_set={fragment_set}, "
            f"func_dis={func_dis}, func_merge={getattr(func_merge, '__name__', str(func_merge))}"
        )
        sm_list = df_raw["SMILES"]
        emb, key_all = tdims_ext.get_representation(
            sm_list,
            radius=radius,
            func_dis=func_dis,
            func_merge=func_merge,
            fragment_set=fragment_set,
            atom_set=True,
            fingerprint_set=True,
            display=True
        )

        df_desc = pd.DataFrame(data=emb, columns=key_all, index=df_raw.index)
        return df_desc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def benchmark_descriptor(X: pd.DataFrame) -> pd.DataFrame:
    """
    Load benchmark embeddings already stored in the input CSV.

    Accepted column patterns
    ------------------------
    - '0', '1', '2', ...
    - '0_name', '1_name', ...
    """
    df_raw = _resolve(X)
    cols = list(df_raw.columns)

    digit_only = [c for c in cols if str(c).isdigit()]
    if digit_only:
        emb_cols = sorted(digit_only, key=lambda c: int(str(c)))
        return df_raw.loc[:, emb_cols]

    digit_prefix = [c for c in cols if re.match(r"^\d+_.+", str(c))]
    if digit_prefix:
        emb_cols = sorted(digit_prefix, key=lambda c: int(str(c).split("_", 1)[0]))
        return df_raw.loc[:, emb_cols]

    raise ValueError(
        "benchmark_descriptor: embedding columns not found. "
        "Expected columns like '0,1,2,...' or '0_{name},1_{name},...'."
    )


def make_benchmark_kwargs_list() -> list[dict[str, Any]]:
    """
    Return a dummy kwargs list for benchmark descriptors.

    Benchmark descriptors do not sweep descriptor-side hyperparameters here,
    but the pipeline expects a list of kwarg dictionaries.
    """
    return [{}]


def make_tdims_kwargs_list(use_all_config: bool) -> list[dict[str, Any]]:
    """
    Enumerate TDiMS hyperparameter candidates.

    Parameters
    ----------
    use_all_config : bool
        If True, use the full Cartesian product of the predefined TDiMS
        hyperparameters. If False, use a manually filtered subset.

    Returns
    -------
    list of dict
        Each dict corresponds to one candidate configuration for
        descriptor__kw_args in GridSearchCV.
    """
    hyperparam_grid = {
        "radius": [1, 2],
        "func_dis": [-2, -1, 1],
        "func_merge": [max, sum],
        "fragment_set": [False, True],
    }

    kwargs_list_all = [
        dict(zip(hyperparam_grid.keys(), value_combo))
        for value_combo in product(*hyperparam_grid.values())
    ]

    blacklist = [(1, sum), (1, min), (-1, min), (-2, min)]
    kwargs_list_selected = [
        kwargs
        for kwargs in kwargs_list_all
        if (kwargs["func_dis"], kwargs["func_merge"]) not in blacklist
    ]
    print(f"TDiMS hyperparameter count (selected): {len(kwargs_list_selected)}")

    return kwargs_list_all if use_all_config else kwargs_list_selected


def make_ml_grid_list(mode: str = "full"):
    """
    Build model-side hyperparameter grids.

    Parameters
    ----------
    mode : {"full", "quick"}, default="full"
        Hyperparameter search mode.

        - "full":
          Full search space intended for the main experiments / paper runs.
          This setting is more computationally expensive.

        - "quick":
          Reduced search space intended for smoke tests, debugging,
          and lightweight example runs. This is faster but does not
          reproduce the full search setting used in the main experiments.

    Returns
    -------
    list of dict
        Hyperparameter grids for GridSearchCV.
    """
    mode = str(mode).lower().strip()
    if mode not in ("full", "quick"):
        raise ValueError("mode must be 'full' or 'quick'")

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

    select_estimator = {
        "select__estimator": [Lasso(random_state=0)],
        "select__estimator__alpha": [0.0001],
        "select__threshold": ["mean"],
    }

    return [
        {
            **select_estimator,
            "model": [Lasso(random_state=0)],
            "model__alpha": alpha_lasso,
        },
        {
            **select_estimator,
            "model": [Ridge(random_state=0)],
            "model__alpha": alpha_ridge,
        },
        {
            **select_estimator,
            "model": [ElasticNet(random_state=0)],
            "model__l1_ratio": en_l1_ratio,
            "model__alpha": alpha_en,
        },
        {
            **select_estimator,
            "model": [RandomForestRegressor(random_state=0)],
            "model__min_samples_split": rf_min_samples_split,
        },
    ]


def build_param_grid(descriptor, descriptor_kwargs_list, ml_grid_list):
    """
    Combine descriptor-side and model-side hyperparameter grids.

    Notes
    -----
    For TDiMS-like descriptors, configurations with func_dis == 1 are handled
    separately because their preprocessing/scaling differs from the others.
    """
    base_desc = {"descriptor": [PandasFunctionTransformer(descriptor)]}
    grids = []

    has_func_dis = any(("func_dis" in (kw or {})) for kw in (descriptor_kwargs_list or []))
    if has_func_dis:
        tdims_dis1 = [kw for kw in descriptor_kwargs_list if kw.get("func_dis") == 1]
        tdims_other = [kw for kw in descriptor_kwargs_list if kw.get("func_dis") != 1]

        if tdims_dis1:
            for mlg in ml_grid_list:
                grids.append(
                    {
                        **base_desc,
                        "descriptor__kw_args": tdims_dis1,
                        "clip_gt1_to0__threshold": [1000.0],
                        "scaler": [StandardScaler(with_mean=False)],
                        **mlg,
                    }
                )

        if tdims_other:
            for mlg in ml_grid_list:
                grids.append(
                    {
                        **base_desc,
                        "descriptor__kw_args": tdims_other,
                        "scaler": ["passthrough"],
                        **mlg,
                    }
                )

        return grids

    for mlg in ml_grid_list:
        grids.append(
            {
                **base_desc,
                "descriptor__kw_args": descriptor_kwargs_list or [{}],
                "to_csr": ["passthrough"],
                "clip_gt1_to0": ["passthrough"],
                "sparse_filter": ["passthrough"],
                "scaler": [StandardScaler(with_mean=True)],
                **mlg,
            }
        )

    return grids


def run_experiment_minimal(
    data_csv_path: str,
    prop: str,
    *,
    descriptor: Callable[..., pd.DataFrame],
    descriptor_kwargs_list: list[dict[str, Any]],
    out_prefix: str,
    out_dir: str = "",
    outer_random_state: int = 0,
    outer_n_repeats: int = 10,
    n_jobs: int = -1,
    verbose: int = 10,
    error_score: Any = "raise",
    use_dummy_label: bool = False,
    save_cv_results: bool = True,
    save_joblib: bool = False,
    save_score: bool = False,
    mode: str = "full",
) -> None:
    """
    Run nested cross-validation for one dataset/property/descriptor setting.

    Outer CV
    --------
    RepeatedKFold(n_splits=3, n_repeats=outer_n_repeats, random_state=outer_random_state)

    Inner CV
    --------
    KFold(n_splits=3, shuffle=True, random_state=0)

    Parameters
    ----------
    mode : {"full", "quick"}, default="full"
        Search-space setting passed to make_ml_grid_list().
        Use "full" for the main experiment setting and "quick" for
        lightweight test runs.

    outer_n_repeats : int, default=10
        Number of repeats for the outer RepeatedKFold.
    """
    df_all = pd.read_csv(data_csv_path)
    n = df_all.shape[0]
    X = pd.DataFrame({"path": [data_csv_path] * n, "row": np.arange(n)})
    y = (np.arange(n) / n) if use_dummy_label else df_all[prop]

    cachedir = tempfile.mkdtemp()
    mem = Memory(location=cachedir, verbose=0)
    print(f"Pipeline cache directory: {cachedir}")

    cv_inner = KFold(n_splits=3, shuffle=True, random_state=0)
    cv_outer = RepeatedKFold(
        n_splits=3,
        n_repeats=outer_n_repeats,
        random_state=outer_random_state,
    )

    ml_grid_list = make_ml_grid_list(mode=mode)
    param_grid = build_param_grid(
        descriptor,
        descriptor_kwargs_list=descriptor_kwargs_list,
        ml_grid_list=ml_grid_list,
    )

    model_prefix = out_prefix
    
    
    out_testscore = Path(f"./{out_dir}/{model_prefix}/scorelist")
    out_testscore.mkdir(parents=True, exist_ok=True)
    if save_joblib:
        out_model_dir = Path(f"./{out_dir}/{model_prefix}/saved_models")
        out_model_dir.mkdir(parents=True, exist_ok=True)
    if save_cv_results:
        out_model_dir_csv = Path(f"./{out_dir}/{model_prefix}/csv")
        out_model_dir_csv.mkdir(parents=True, exist_ok=True)
    
    splits_outer = list(cv_outer.split(X, y))
    n_folds = len(splits_outer)
    outer_scores: list[Optional[float]] = [None] * n_folds

    for fold_i, (tr_idx, te_idx) in enumerate(splits_outer):
        fold_tag = f"fold{fold_i}"
        print(f"fold{fold_i}")
        
        if save_score:
            fold_score_npy = out_testscore / f"{model_prefix}_seed{outer_random_state}_{fold_tag}_outer_score.npy"

            if fold_score_npy.exists():
                print(f"[LOAD] {fold_tag}: {fold_score_npy}")
                outer_scores[fold_i] = float(np.load(fold_score_npy))
                continue

        X_tr, y_tr = X.iloc[tr_idx], np.asarray(y)[tr_idx]

        gscv = GridSearchCV(
            estimator=make_pipeline_template(memory_obj=mem),
            param_grid=param_grid,
            cv=cv_inner,
            n_jobs=n_jobs,
            error_score=error_score,
            refit=True,
            return_train_score=False,
            verbose=verbose,
        )
        gscv.fit(X_tr, y_tr)

        if save_cv_results:
            fold_cv_csv = out_model_dir_csv / f"{model_prefix}_seed{outer_random_state}_{fold_tag}_cv_results.csv"
            pd.DataFrame(gscv.cv_results_).to_csv(fold_cv_csv, index=False)

        if save_joblib:
            import joblib
            fold_joblib = out_model_dir / f"{model_prefix}_seed{outer_random_state}_{fold_tag}.joblib"
            joblib.dump(gscv, fold_joblib, compress=3)

        X_te, y_te = X.iloc[te_idx], np.asarray(y)[te_idx]
        outer_score = float(gscv.best_estimator_.score(X_te, y_te))
        
        if save_score:
            np.save(fold_score_npy, np.array(outer_score, dtype=float))
        outer_scores[fold_i] = outer_score

    pd.DataFrame({"fold": list(range(n_folds)), "outer_r2": outer_scores}).to_csv(
        out_testscore / f"{model_prefix}_seed{outer_random_state}_outer_scores_all.csv",
        index=False,
    )
    
    print(f'Saved files to: {model_prefix}_seed{outer_random_state}_outer_scores_all.csv')


def main(
    *,
    database: str,
    prop: str,
    desc_name: str = "TDiMS",
    outer_random_state: int = 0,
    outer_n_repeats: int = 10,
    n_jobs: int = -1,
    out_dir: str = "",
    etc: str = "",
    save_cv_results: bool = True,
    save_joblib: bool = False,
    save_score=False,
    mode: str = "full",
    verbose: int = 0
) -> None:
    """
    Entry point for running one experiment configuration.

    Parameters
    ----------
    mode : {"full", "quick"}, default="full"
        Search-space setting.
        - "full": main experimental setting
        - "quick": reduced setting for fast tests/examples

    outer_n_repeats : int, default=10
        Number of repeats for the outer RepeatedKFold.
    """
    descriptor_configs = {
        "TDiMS": {
            "descriptor": tdims_descriptor,
            "descriptor_kwargs_list": make_tdims_kwargs_list(use_all_config=False),
        },
        "AtomPair": {"descriptor": benchmark_descriptor, "descriptor_kwargs_list": make_benchmark_kwargs_list()},
        "mordred": {"descriptor": benchmark_descriptor, "descriptor_kwargs_list": make_benchmark_kwargs_list()},
        "mordred3D": {"descriptor": benchmark_descriptor, "descriptor_kwargs_list": make_benchmark_kwargs_list()},
        "MAP4": {"descriptor": benchmark_descriptor, "descriptor_kwargs_list": make_benchmark_kwargs_list()},
        "molformer": {"descriptor": benchmark_descriptor, "descriptor_kwargs_list": make_benchmark_kwargs_list()},
        "MolCLR": {"descriptor": benchmark_descriptor, "descriptor_kwargs_list": make_benchmark_kwargs_list()},
    }

    if desc_name not in descriptor_configs:
        raise ValueError(f"Unknown desc_name: {desc_name}. Available: {list(descriptor_configs)}")

    base = Path(database).stem
    out_prefix = f"{base}_{prop}_{desc_name}"
    if etc:
        out_prefix = f"{out_prefix}_{etc}"

    cfg = descriptor_configs[desc_name]

    run_experiment_minimal(
        data_csv_path=database,
        prop=prop,
        descriptor=cfg["descriptor"],
        descriptor_kwargs_list=cfg["descriptor_kwargs_list"],
        out_prefix=out_prefix,
        out_dir=out_dir,
        outer_random_state=outer_random_state,
        outer_n_repeats=outer_n_repeats,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise",
        use_dummy_label=False,
        save_cv_results=save_cv_results,
        save_joblib=save_joblib,
        save_score=save_score,
        mode=mode,
    )


if __name__ == "__main__":
    main(
        database="./data/cmpCl3_1000.csv",
        prop="Abs",
        desc_name="TDiMS",
        outer_random_state=0,
        outer_n_repeats=1,
        n_jobs=1,
        out_dir="Table1",
        etc="Table1_test_v2",
        save_cv_results=True,
        save_joblib=False,
        save_score=False,
        mode="quick",
        verbose=0
    )