from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_retail_daily_primary


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_retail_daily_primary,
                inputs=["transactions_clean", "stores_clean", "calendar_clean"],
                outputs="retail_daily_primary",
                name="build_retail_daily_primary_node",
            ),
        ]
    )
