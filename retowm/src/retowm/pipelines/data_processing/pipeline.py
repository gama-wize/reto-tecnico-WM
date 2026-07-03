from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_calendar, clean_stores, clean_transactions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_transactions,
                inputs=["transactions", "parameters"],
                outputs="transactions_clean",
                name="clean_transactions_node",
            ),
            node(
                func=clean_stores,
                inputs=["stores", "parameters"],
                outputs="stores_clean",
                name="clean_stores_node",
            ),
            node(
                func=clean_calendar,
                inputs=["calendar", "parameters"],
                outputs="calendar_clean",
                name="clean_calendar_node",
            ),
        ]
    )
