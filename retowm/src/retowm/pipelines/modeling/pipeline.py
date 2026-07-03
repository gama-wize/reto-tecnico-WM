from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_segments, train_and_evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_and_evaluate_model,
                inputs={
                    "train_df": "model_input_train",
                    "test_df": "model_input_test",
                    "parameters": "parameters",
                },
                outputs=["trained_model", "test_predictions", "model_comparison", "model_metrics"],
                name="train_and_evaluate_model",
            ),
            node(
                func=evaluate_segments,
                inputs={
                    "test_predictions": "test_predictions",
                    "parameters": "parameters",
                },
                outputs="segment_metrics",
                name="evaluate_segments",
            ),
        ]
    )
