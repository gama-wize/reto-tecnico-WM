from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_retail_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_retail_features,
                inputs=["retail_daily_primary", "parameters"],
                outputs="retail_features",
                name="build_retail_features_node",
            ),
        ]
    )
