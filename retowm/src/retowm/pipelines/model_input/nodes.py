import pandas as pd


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Raise ValueError if any required column is missing from df."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing}"
        )


def _deduplicate_preserve_order(columns: list[str]) -> list[str]:
    """Return list with duplicates removed while preserving original order."""
    seen = set()
    result = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            result.append(col)
    return result


def _validate_no_leakage_columns(
    df: pd.DataFrame,
    leakage_columns: list[str],
    target_column: str,
    dataset_name: str,
) -> None:
    """Raise ValueError if any leakage column (excluding target) is present in df."""
    forbidden = [c for c in leakage_columns if c in df.columns and c != target_column]
    if forbidden:
        raise ValueError(
            f"Dataset '{dataset_name}' contains leakage columns: {forbidden}"
        )


def _get_engineered_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    """Return all lag and rolling feature columns derived from the given target."""
    return [
        c for c in df.columns
        if c.startswith(f"{target_column}_lag_") or c.startswith(f"{target_column}_rolling_")
    ]


def build_model_inputs(
    retail_features: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the feature dataset into temporal train and test sets, validating for leakage.

    Selects the final feature columns, drops rows with nulls in lag/rolling features,
    and validates that the train and test periods are strictly non-overlapping.

    Args:
        retail_features: Feature-engineered dataset from the feature_engineering pipeline.
        parameters: Kedro parameters dict with forecasting config (train/test date bounds,
            target_column, leakage_columns, safe_base_feature_columns, entity_columns, etc.).

    Returns:
        Tuple of (train_df, test_df), each sorted by date and entity columns.

    Raises:
        ValueError: On missing columns, leakage column presence, empty splits, or
            temporal overlap between train and test.
    """
    forecasting_params = parameters["forecasting"]

    target_column = forecasting_params["target_column"]
    date_column = forecasting_params["date_column"]
    original_date_column = forecasting_params["original_date_column"]
    entity_columns = forecasting_params["entity_columns"]
    train_start_date = forecasting_params["train_start_date"]
    train_end_date = forecasting_params["train_end_date"]
    test_start_date = forecasting_params["test_start_date"]
    test_end_date = forecasting_params["test_end_date"]
    safe_base_feature_columns = forecasting_params["safe_base_feature_columns"]
    leakage_columns = forecasting_params["leakage_columns"]

    required = (
        [target_column, date_column, original_date_column]
        + entity_columns
        + safe_base_feature_columns
    )
    _validate_required_columns(retail_features, required, "retail_features")

    df = retail_features.copy()
    _validate_no_leakage_columns(df, leakage_columns, target_column, "retail_features")

    engineered_columns = _get_engineered_feature_columns(df, target_column)

    safe_cols_present = [c for c in safe_base_feature_columns if c in df.columns]
    leakage_set = set(leakage_columns) - {target_column}

    final_columns = _deduplicate_preserve_order(
        [original_date_column, date_column]
        + entity_columns
        + [target_column]
        + [c for c in safe_cols_present if c not in leakage_set]
        + engineered_columns
    )
    df = df[final_columns]

    lag_and_rolling_cols = [
        c for c in engineered_columns if c in df.columns
    ]
    df = df.dropna(subset=lag_and_rolling_cols)

    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors="raise")

    train_start = pd.Timestamp(train_start_date)
    train_end = pd.Timestamp(train_end_date)
    test_start = pd.Timestamp(test_start_date)
    test_end = pd.Timestamp(test_end_date)

    train = df[(df[date_column] >= train_start) & (df[date_column] <= train_end)].copy()
    test = df[(df[date_column] >= test_start) & (df[date_column] <= test_end)].copy()

    if train.empty:
        raise ValueError("Train split is empty. Check train_start_date / train_end_date.")
    if test.empty:
        raise ValueError("Test split is empty. Check test_start_date / test_end_date.")
    if train[date_column].max() >= test[date_column].min():
        raise ValueError("Temporal leakage: train max date >= test min date.")

    for split_name, split_df in [("model_input_train", train), ("model_input_test", test)]:
        if target_column not in split_df.columns:
            raise ValueError(f"Target '{target_column}' missing from {split_name}.")
        missing_eng = [c for c in lag_and_rolling_cols if c not in split_df.columns]
        if missing_eng:
            raise ValueError(f"{split_name} is missing engineered columns: {missing_eng}")
        null_eng = [c for c in lag_and_rolling_cols if split_df[c].isna().any()]
        if null_eng:
            raise ValueError(f"{split_name} has nulls in lag/rolling columns: {null_eng}")

    sort_cols = [date_column] + entity_columns
    train = train.sort_values(sort_cols).reset_index(drop=True)
    test = test.sort_values(sort_cols).reset_index(drop=True)

    return train, test
