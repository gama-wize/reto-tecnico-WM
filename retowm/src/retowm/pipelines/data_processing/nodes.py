import pandas as pd


def apply_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    missing = set(schema.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df.astype(schema)


def clean_transactions(transactions: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    schema = parameters["schema"]["transactions"]
    df = apply_schema(transactions, schema)
    return df.sort_values(["date", "store_id", "category"]).reset_index(drop=True)


def clean_stores(stores: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    schema = parameters["schema"]["stores"]
    df = apply_schema(stores, schema)
    return df.sort_values("store_id").reset_index(drop=True)


def clean_calendar(calendar: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    schema = parameters["schema"]["calendar"]
    df = apply_schema(calendar, schema)
    return df.sort_values("date").reset_index(drop=True)
