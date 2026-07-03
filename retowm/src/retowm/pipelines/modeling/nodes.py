import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _to_numpy_compatible(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.StringDtype):
            # StringDtype uses pd.NA; convert to object with None so sklearn handles it
            df[col] = df[col].to_numpy(dtype=object, na_value=None)
        elif hasattr(dtype, "numpy_dtype"):
            # All other pandas extension types (BooleanDtype, Int64Dtype, etc.)
            np_dtype = dtype.numpy_dtype
            if np_dtype == np.dtype("bool"):
                df[col] = df[col].astype("float64")
            else:
                df[col] = df[col].astype("float64")
        elif df[col].dtype == np.dtype("bool"):
            # Plain numpy bool — cast to float so SimpleImputer handles it
            df[col] = df[col].astype("float64")
    return df


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    contributions = np.where(denominator > 0, 2 * np.abs(y_pred - y_true) / denominator, 0.0)
    return float(100 * np.mean(contributions))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    smape = _smape(y_true, y_pred)
    mean_abs_true = float(np.mean(np.abs(y_true)))
    relative_mae = float(mae / mean_abs_true) if mean_abs_true > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "smape": smape, "relative_mae": relative_mae}


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    parameters: dict,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, dict]:
    modeling_params = parameters["modeling"]

    target_column = modeling_params["target_column"]
    baseline_column = modeling_params["baseline_column"]
    id_columns = modeling_params["id_columns"]
    categorical_columns = modeling_params["categorical_columns"]
    excluded_feature_columns = modeling_params["excluded_feature_columns"]
    segment_columns = modeling_params["segment_columns"]
    lightgbm_params = modeling_params["lightgbm_params"]
    evaluation = modeling_params["evaluation"]

    if train_df.empty:
        raise ValueError("train_df is empty.")
    if test_df.empty:
        raise ValueError("test_df is empty.")
    if target_column not in train_df.columns:
        raise ValueError(f"Target '{target_column}' not in train_df.")
    if target_column not in test_df.columns:
        raise ValueError(f"Target '{target_column}' not in test_df.")
    if baseline_column not in test_df.columns:
        raise ValueError(f"Baseline column '{baseline_column}' not in test_df.")

    exclude_set = set(excluded_feature_columns)

    feature_columns = [
        c for c in train_df.columns
        if c not in exclude_set and c in test_df.columns
    ]

    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions.")

    y_train = train_df[target_column].to_numpy()
    y_test = test_df[target_column].to_numpy()

    if pd.isnull(y_train).any():
        raise ValueError("y_train contains null values.")
    if pd.isnull(y_test).any():
        raise ValueError("y_test contains null values.")

    categorical_columns_present = [c for c in categorical_columns if c in feature_columns]
    numeric_columns_present = [c for c in feature_columns if c not in categorical_columns_present]

    X_train = _to_numpy_compatible(train_df[feature_columns])
    X_test = _to_numpy_compatible(test_df[feature_columns])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns_present),
            ("num", numeric_transformer, numeric_columns_present),
        ],
        remainder="drop",
    )

    lgbm_kwargs = {k: v for k, v in lightgbm_params.items() if k != "objective"}
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(**lgbm_kwargs, objective="regression")),
    ])

    model_pipeline.fit(X_train, y_train)
    lgbm_pred = model_pipeline.predict(X_test)
    baseline_pred = test_df[baseline_column].to_numpy()

    baseline_metrics = _regression_metrics(y_test, baseline_pred)
    lgbm_metrics = _regression_metrics(y_test, lgbm_pred)

    model_comparison = pd.DataFrame([
        {"model": "baseline_lag_7", **baseline_metrics},
        {"model": "lightgbm", **lgbm_metrics},
    ]).sort_values("mae").reset_index(drop=True)

    best_model = model_comparison.iloc[0]["model"]

    model_metrics = {
        "target_column": target_column,
        "baseline_column": baseline_column,
        "validation_strategy": evaluation["validation_strategy"],
        "train_period": evaluation["train_period"],
        "test_period": evaluation["test_period"],
        "no_cross_validation_reason": evaluation["no_cross_validation_reason"],
        "metrics_by_model": {
            "baseline_lag_7": baseline_metrics,
            "lightgbm": lgbm_metrics,
        },
        "best_model_by_mae": best_model,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_features": len(feature_columns),
        "categorical_features": categorical_columns_present,
        "numeric_features": numeric_columns_present,
        "lightgbm_params": lightgbm_params,
    }

    id_cols_present = [c for c in id_columns if c in test_df.columns]
    seg_cols_present = [c for c in segment_columns if c in test_df.columns]
    extra_cols = [c for c in seg_cols_present if c not in id_cols_present]

    test_predictions = test_df[id_cols_present + extra_cols].copy()
    test_predictions["y_true"] = y_test
    test_predictions["baseline_lag_7_pred"] = baseline_pred
    test_predictions["lightgbm_pred"] = lgbm_pred
    test_predictions["baseline_lag_7_error"] = baseline_pred - y_test
    test_predictions["lightgbm_error"] = lgbm_pred - y_test
    test_predictions["baseline_lag_7_abs_error"] = np.abs(baseline_pred - y_test)
    test_predictions["lightgbm_abs_error"] = np.abs(lgbm_pred - y_test)

    return model_pipeline, test_predictions, model_comparison, model_metrics


def evaluate_segments(
    test_predictions: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    if "y_true" not in test_predictions.columns:
        raise ValueError("test_predictions is missing 'y_true'.")
    if "baseline_lag_7_pred" not in test_predictions.columns:
        raise ValueError("test_predictions is missing 'baseline_lag_7_pred'.")
    if "lightgbm_pred" not in test_predictions.columns:
        raise ValueError("test_predictions is missing 'lightgbm_pred'.")

    segment_columns = parameters["modeling"]["segment_columns"]
    available = [c for c in segment_columns if c in test_predictions.columns]

    output_columns = ["segment_column", "segment_value", "model", "n_rows",
                      "mae", "rmse", "smape", "relative_mae"]

    if not available:
        return pd.DataFrame(columns=output_columns)

    models = {
        "baseline_lag_7": "baseline_lag_7_pred",
        "lightgbm": "lightgbm_pred",
    }

    rows = []
    for seg_col in available:
        for model_name, pred_col in models.items():
            subset = test_predictions[[seg_col, "y_true", pred_col]].dropna()
            for seg_val, group in subset.groupby(seg_col):
                metrics = _regression_metrics(
                    group["y_true"].to_numpy(),
                    group[pred_col].to_numpy(),
                )
                rows.append({
                    "segment_column": seg_col,
                    "segment_value": str(seg_val),
                    "model": model_name,
                    "n_rows": len(group),
                    **metrics,
                })

    result = pd.DataFrame(rows, columns=output_columns)
    return result.sort_values(["segment_column", "segment_value", "model"]).reset_index(drop=True)
