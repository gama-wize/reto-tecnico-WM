import pandas as pd


def apply_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Enforce exact column set and dtypes defined in schema; reject extra or missing columns.

    Args:
        df: Input DataFrame to validate and cast.
        schema: Mapping of column name to target dtype string (e.g. ``{"date": "str", "amount": "Float64"}``).

    Returns:
        A copy of df with columns reordered and cast to the schema dtypes.

    Raises:
        ValueError: If any schema column is missing from df, or if df has columns not in schema.
    """
    df = df.copy()
    missing = sorted(set(schema.keys()) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    extra = sorted(set(df.columns) - set(schema.keys()))
    if extra:
        raise ValueError(f"Unexpected columns not in schema: {extra}")
    return df.astype(schema)[list(schema.keys())]


def clean_transactions(transactions: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Apply schema enforcement to the raw transactions dataset and sort by date, store, category.

    Args:
        transactions: Raw transactions DataFrame loaded from CSV.
        parameters: Kedro parameters dict containing ``schema["transactions"]``.

    Returns:
        Schema-typed transactions DataFrame sorted by ``["date", "store_id", "category"]``.
    """
    schema = parameters["schema"]["transactions"]
    df = apply_schema(transactions, schema)
    return df.sort_values(["date", "store_id", "category"]).reset_index(drop=True)


def clean_stores(stores: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Apply schema enforcement to the raw stores dataset and sort by store_id.

    Args:
        stores: Raw stores DataFrame loaded from CSV.
        parameters: Kedro parameters dict containing ``schema["stores"]``.

    Returns:
        Schema-typed stores DataFrame sorted by ``store_id``.
    """
    schema = parameters["schema"]["stores"]
    df = apply_schema(stores, schema)
    return df.sort_values("store_id").reset_index(drop=True)


def clean_calendar(calendar: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Apply schema enforcement to the raw calendar dataset and sort by date.

    Args:
        calendar: Raw calendar DataFrame loaded from CSV.
        parameters: Kedro parameters dict containing ``schema["calendar"]``.

    Returns:
        Schema-typed calendar DataFrame sorted by ``date``.
    """
    schema = parameters["schema"]["calendar"]
    df = apply_schema(calendar, schema)
    return df.sort_values("date").reset_index(drop=True)
