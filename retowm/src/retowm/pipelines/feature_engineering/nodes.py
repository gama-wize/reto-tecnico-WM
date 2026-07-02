import pandas as pd


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing}"
        )


def _deduplicate_preserve_order(columns: list[str]) -> list[str]:
    seen = set()
    result = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            result.append(col)
    return result


def build_retail_features(
    retail_daily_primary: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    forecasting_params = parameters["forecasting"]

    target_column = forecasting_params["target_column"]
    date_column = forecasting_params["date_column"]
    original_date_column = forecasting_params["original_date_column"]
    entity_columns = forecasting_params["entity_columns"]
    lags = forecasting_params["lags"]
    rolling_windows = forecasting_params["rolling_windows"]
    rolling_statistics = forecasting_params["rolling_statistics"]
    safe_base_feature_columns = forecasting_params["safe_base_feature_columns"]
    leakage_columns = forecasting_params["leakage_columns"]

    required = [target_column, date_column, original_date_column] + entity_columns + safe_base_feature_columns
    _validate_required_columns(retail_daily_primary, required, "retail_daily_primary")

    df = retail_daily_primary.copy()
    df = df.sort_values(entity_columns + [date_column]).reset_index(drop=True)

    for lag in lags:
        col_name = f"{target_column}_lag_{lag}"
        df[col_name] = df.groupby(entity_columns)[target_column].shift(lag)

    lag_feature_columns = [f"{target_column}_lag_{lag}" for lag in lags]

    rolling_feature_columns = []
    for window in rolling_windows:
        if "mean" in rolling_statistics:
            col_name = f"{target_column}_rolling_mean_{window}"
            df[col_name] = df.groupby(entity_columns)[target_column].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
            )
            rolling_feature_columns.append(col_name)
        if "std" in rolling_statistics:
            col_name = f"{target_column}_rolling_std_{window}"
            df[col_name] = df.groupby(entity_columns)[target_column].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=2).std()
            )
            rolling_feature_columns.append(col_name)

    columns_to_drop = [
        c for c in leakage_columns
        if c in df.columns and c != target_column
    ]
    df = df.drop(columns=columns_to_drop)

    base_columns = [original_date_column, date_column] + entity_columns + [target_column]
    safe_cols_present = [c for c in safe_base_feature_columns if c in df.columns]

    final_columns = _deduplicate_preserve_order(
        base_columns + safe_cols_present + lag_feature_columns + rolling_feature_columns
    )

    return (
        df[final_columns]
        .sort_values([date_column, "store_id", "category"])
        .reset_index(drop=True)
    )
