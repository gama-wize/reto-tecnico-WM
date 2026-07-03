from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_model_inputs


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_model_inputs,
                inputs=["retail_features", "parameters"],
                outputs=["model_input_train", "model_input_test"],
                name="build_model_inputs_node",
            ),
        ]
    )
