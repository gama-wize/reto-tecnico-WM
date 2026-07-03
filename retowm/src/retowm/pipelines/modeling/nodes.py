import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _to_numpy_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas extension dtypes to numpy-compatible equivalents for sklearn."""
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.StringDtype):
            # StringDtype uses pd.NA; convert to object with None so sklearn handles it
            df[col] = df[col].to_numpy(dtype=object, na_value=None)
        elif hasattr(dtype, "numpy_dtype"):
            # All other pandas extension types (BooleanDtype, Int64Dtype, etc.)
            df[col] = df[col].astype("float64")
        elif df[col].dtype == np.dtype("bool"):
            # Plain numpy bool — cast to float so SimpleImputer handles it
            df[col] = df[col].astype("float64")
    return df


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute symmetric mean absolute percentage error (sMAPE) as a percentage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    contributions = np.where(denominator > 0, 2 * np.abs(y_pred - y_true) / denominator, 0.0)
    return float(100 * np.mean(contributions))


def _baseline_model_name(baseline_column: str, target_column: str) -> str:
    """Derive the baseline model label from the baseline column name (e.g. ``'baseline_lag_7'``)."""
    prefix = target_column + "_"
    suffix = baseline_column[len(prefix):] if baseline_column.startswith(prefix) else baseline_column
    return f"baseline_{suffix}"


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with mae, rmse, smape, and relative_mae for a set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    smape = _smape(y_true, y_pred)
    mean_abs_true = float(np.mean(np.abs(y_true)))
    relative_mae = float(mae / mean_abs_true) if mean_abs_true > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "smape": smape, "relative_mae": relative_mae}


def train_and_evaluate_model(  # noqa: PLR0915
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    parameters: dict,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, dict]:
    """Train a LightGBM pipeline with sklearn preprocessing and evaluate against a lag baseline.

    Builds a ColumnTransformer with OneHotEncoder for categoricals and SimpleImputer for
    numerics, fits LGBMRegressor on the train set, and produces predictions, metrics, and
    a comparison table for both the model and the baseline.

    Args:
        train_df: Training split from the model_input pipeline.
        test_df: Test split from the model_input pipeline. Must contain the baseline column.
        parameters: Kedro parameters dict with modeling config (target_column, baseline_column,
            categorical_columns, excluded_feature_columns, lightgbm_params, etc.).

    Returns:
        Tuple of:
        - Trained sklearn Pipeline (preprocessor + LGBMRegressor).
        - test_predictions DataFrame with y_true, model predictions, errors, and abs errors.
        - model_comparison DataFrame with MAE, RMSE, sMAPE, relative_MAE per model.
        - model_metrics dict with training metadata, hyperparameters, and metrics by model.

    Raises:
        ValueError: If required columns are missing, target contains nulls, or no features remain.
    """
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

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(**lightgbm_params)),
    ])
    model_pipeline.set_output(transform="pandas")

    model_pipeline.fit(X_train, y_train)
    lgbm_pred = model_pipeline.predict(X_test)
    baseline_pred = test_df[baseline_column].to_numpy()

    bname = _baseline_model_name(baseline_column, target_column)

    baseline_metrics = _regression_metrics(y_test, baseline_pred)
    lgbm_metrics = _regression_metrics(y_test, lgbm_pred)

    model_comparison = pd.DataFrame([
        {"model": bname, **baseline_metrics},
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
            bname: baseline_metrics,
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
    test_predictions[f"{bname}_pred"] = baseline_pred
    test_predictions["lightgbm_pred"] = lgbm_pred
    test_predictions[f"{bname}_error"] = baseline_pred - y_test
    test_predictions["lightgbm_error"] = lgbm_pred - y_test
    test_predictions[f"{bname}_abs_error"] = np.abs(baseline_pred - y_test)
    test_predictions["lightgbm_abs_error"] = np.abs(lgbm_pred - y_test)

    return model_pipeline, test_predictions, model_comparison, model_metrics


def evaluate_segments(
    test_predictions: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Compute MAE, RMSE, sMAPE, and relative MAE per model broken down by segment column.

    Iterates over all segment columns defined in parameters and computes per-segment
    metrics for both the baseline and LightGBM predictions.

    Args:
        test_predictions: Output of train_and_evaluate_model containing y_true,
            baseline predictions, lightgbm predictions, and segment columns.
        parameters: Kedro parameters dict with modeling config (segment_columns,
            baseline_column, target_column).

    Returns:
        Long-format DataFrame with columns: segment_column, segment_value, model, n_rows,
        mae, rmse, smape, relative_mae. Returns empty DataFrame if no segment columns
        are available in test_predictions.

    Raises:
        ValueError: If y_true or prediction columns are missing from test_predictions.
    """
    modeling_params = parameters["modeling"]
    bname = _baseline_model_name(modeling_params["baseline_column"], modeling_params["target_column"])

    if "y_true" not in test_predictions.columns:
        raise ValueError("test_predictions is missing 'y_true'.")
    if f"{bname}_pred" not in test_predictions.columns:
        raise ValueError(f"test_predictions is missing '{bname}_pred'.")
    if "lightgbm_pred" not in test_predictions.columns:
        raise ValueError("test_predictions is missing 'lightgbm_pred'.")

    segment_columns = modeling_params["segment_columns"]
    available = [c for c in segment_columns if c in test_predictions.columns]

    output_columns = ["segment_column", "segment_value", "model", "n_rows",
                      "mae", "rmse", "smape", "relative_mae"]

    if not available:
        return pd.DataFrame(columns=output_columns)

    models = {
        bname: f"{bname}_pred",
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


def _map_original_feature(transformed_name: str, categorical_columns: list[str]) -> str:
    """Map a ColumnTransformer-prefixed feature name back to its original source column."""
    if transformed_name.startswith("num__"):
        return transformed_name[len("num__"):]
    if transformed_name.startswith("cat__"):
        remainder = transformed_name[len("cat__"):]
        matches = [c for c in categorical_columns if remainder.startswith(c + "_")]
        if matches:
            return max(matches, key=len)
        return remainder
    return transformed_name


def extract_feature_importance(
    trained_model: Pipeline,
    parameters: dict,
) -> pd.DataFrame:
    """Extract split and gain feature importance for both encoded and original features.

    Retrieves importances from the LightGBM booster (split and gain), maps encoded
    feature names back to their original source columns, and aggregates importance
    by original feature via sum. Returns all four combinations
    (encoded/original × split/gain) in a single long-format DataFrame.

    Args:
        trained_model: Fitted sklearn Pipeline with named steps ``'preprocessor'``
            (ColumnTransformer) and ``'regressor'`` (LGBMRegressor).
        parameters: Kedro parameters dict with modeling config (categorical_columns).

    Returns:
        Long-format DataFrame with columns: importance_level, feature, original_feature,
        importance_type, importance, importance_pct, rank. Sorted by importance_level,
        importance_type, rank, and feature.

    Raises:
        ValueError: If the pipeline is missing expected named steps or if feature name
            and importance array lengths do not match.
    """
    if not hasattr(trained_model, "named_steps"):
        raise ValueError("trained_model does not have named_steps.")
    if "preprocessor" not in trained_model.named_steps:
        raise ValueError("trained_model is missing named_step 'preprocessor'.")
    if "regressor" not in trained_model.named_steps:
        raise ValueError("trained_model is missing named_step 'regressor'.")

    preprocessor = trained_model.named_steps["preprocessor"]
    regressor = trained_model.named_steps["regressor"]

    if not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError("preprocessor does not support get_feature_names_out().")
    if not hasattr(regressor, "feature_importances_"):
        raise ValueError("regressor does not expose feature_importances_.")
    if not hasattr(regressor, "booster_"):
        raise ValueError("regressor does not expose booster_.")

    feature_names = list(preprocessor.get_feature_names_out())
    split_importances = regressor.feature_importances_
    gain_importances = regressor.booster_.feature_importance(importance_type="gain")

    if len(feature_names) != len(split_importances):
        raise ValueError(
            f"Feature names ({len(feature_names)}) do not match split importances ({len(split_importances)})."
        )
    if len(feature_names) != len(gain_importances):
        raise ValueError(
            f"Feature names ({len(feature_names)}) do not match gain importances ({len(gain_importances)})."
        )

    categorical_columns = parameters["modeling"]["categorical_columns"]

    original_features = [
        _map_original_feature(name, categorical_columns) for name in feature_names
    ]

    output_columns = [
        "importance_level", "feature", "original_feature",
        "importance_type", "importance", "importance_pct", "rank",
    ]

    def _build_encoded_rows(imp_values: np.ndarray, imp_type: str) -> pd.DataFrame:
        """Build importance rows for encoded (post-transform) features."""
        total = float(imp_values.sum())
        rows = []
        for name, orig, val in zip(feature_names, original_features, imp_values):
            rows.append({
                "importance_level": "encoded_feature",
                "feature": name,
                "original_feature": orig,
                "importance_type": imp_type,
                "importance": float(val),
                "importance_pct": float(val / total) if total > 0 else 0.0,
            })
        df = pd.DataFrame(rows)
        df["rank"] = df["importance"].rank(method="min", ascending=False).astype(int)
        return df[output_columns]

    def _build_original_rows(imp_values: np.ndarray, imp_type: str) -> pd.DataFrame:
        """Build importance rows aggregated by original source feature."""
        tmp = pd.DataFrame({
            "original_feature": original_features,
            "importance": imp_values.astype(float),
        })
        agg = tmp.groupby("original_feature", sort=False)["importance"].sum().reset_index()
        total = float(agg["importance"].sum())
        agg["importance_pct"] = agg["importance"].apply(
            lambda v: float(v / total) if total > 0 else 0.0
        )
        agg["rank"] = agg["importance"].rank(method="min", ascending=False).astype(int)
        agg["importance_level"] = "original_feature"
        agg["feature"] = agg["original_feature"]
        agg["importance_type"] = imp_type
        return agg[output_columns]

    parts = [
        _build_encoded_rows(split_importances, "split"),
        _build_encoded_rows(gain_importances, "gain"),
        _build_original_rows(split_importances, "split"),
        _build_original_rows(gain_importances, "gain"),
    ]

    result = pd.concat(parts, ignore_index=True)[output_columns]
    return result.sort_values(
        ["importance_level", "importance_type", "rank", "feature"]
    ).reset_index(drop=True)
