import pandas as pd

PRIMARY_KEY = ["date", "store_id", "category"]


def validate_unique_key(
    df: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
) -> None:
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

    validate_unique_key(primary, PRIMARY_KEY, "retail_daily_primary")

    primary["date_dt"] = pd.to_datetime(primary["date"], format="%Y-%m-%d", errors="raise")

    return primary.sort_values(["date_dt", "store_id", "category"]).reset_index(drop=True)
