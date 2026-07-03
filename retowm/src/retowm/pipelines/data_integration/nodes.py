import pandas as pd

PRIMARY_KEY = ["date", "store_id", "category"]


def validate_unique_key(
    df: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
) -> None:
    """Assert that the combination of key_columns is unique across all rows in df.

    Args:
        df: DataFrame to validate.
        key_columns: Column names that together form the expected unique key.
        dataset_name: Label used in the error message to identify the dataset.

    Raises:
        ValueError: If any duplicate rows are found for the given key columns.
    """
    n_duplicates = df.duplicated(subset=key_columns).sum()
    if n_duplicates > 0:
        raise ValueError(
            f"Dataset '{dataset_name}' has {n_duplicates} duplicate rows "
            f"for key columns {key_columns}."
        )


def build_retail_daily_primary(
    transactions_clean: pd.DataFrame,
    stores_clean: pd.DataFrame,
    calendar_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Join cleaned transactions with store attributes and calendar variables into the analytical dataset.

    Validates uniqueness of primary keys before and after joining, enforces row count
    integrity, and checks for unmatched keys on non-nullable source columns.

    Args:
        transactions_clean: Schema-typed transactions at date × store_id × category grain.
        stores_clean: Schema-typed store attributes, one row per store_id.
        calendar_clean: Schema-typed calendar variables, one row per date.

    Returns:
        Joined analytical DataFrame sorted by date_dt, store_id, category, with a
        parsed ``date_dt`` datetime column added.

    Raises:
        ValueError: On duplicate keys, row count change after join, or unmatched foreign keys.
    """
    validate_unique_key(transactions_clean, PRIMARY_KEY, "transactions_clean")
    validate_unique_key(stores_clean, ["store_id"], "stores_clean")
    validate_unique_key(calendar_clean, ["date"], "calendar_clean")

    initial_rows = len(transactions_clean)

    primary = transactions_clean.copy()

    primary = primary.merge(stores_clean, on="store_id", how="left", validate="many_to_one")
    primary = primary.merge(calendar_clean, on="date", how="left", validate="many_to_one")

    if len(primary) != initial_rows:
        raise ValueError(
            f"Row count changed after joins: expected {initial_rows}, got {len(primary)}."
        )

    store_required = [c for c in stores_clean.columns if c != "store_id" and stores_clean[c].notna().all()]
    cal_required = [c for c in calendar_clean.columns if c != "date" and calendar_clean[c].notna().all()]
    null_store = int(primary[store_required].isna().any(axis=1).sum()) if store_required else 0
    null_cal = int(primary[cal_required].isna().any(axis=1).sum()) if cal_required else 0
    if null_store > 0:
        raise ValueError(f"{null_store} rows have unmatched store_id after join with stores.")
    if null_cal > 0:
        raise ValueError(f"{null_cal} rows have unmatched date after join with calendar.")

    validate_unique_key(primary, PRIMARY_KEY, "retail_daily_primary")

    primary["date_dt"] = pd.to_datetime(primary["date"], format="%Y-%m-%d", errors="raise")

    return primary.sort_values(["date_dt", "store_id", "category"]).reset_index(drop=True)
