import time

from kedro.framework.hooks import hook_impl

_W = 60


def _banner(message: str) -> None:
    line = "#" * _W
    padded = f"#   {message:<{_W - 5}}#"
    print(f"\n{line}\n{padded}\n{line}\n", flush=True)  # noqa: T201


class PipelineLoggingHooks:
    """Print pipeline start and completion banners during kedro run."""

    def __init__(self):
        self._current_pipeline: str | None = None
        self._start_time: float | None = None

    @hook_impl
    def before_node_run(self, node, catalog, inputs, is_async, run_id):
        pipeline_tag = next(iter(node.tags), None)
        if pipeline_tag and pipeline_tag != self._current_pipeline:
            if self._current_pipeline is not None:
                elapsed = time.time() - self._start_time
                _banner(f"DONE  {self._current_pipeline}  ({elapsed:.1f} s)")
            _banner(f"START {pipeline_tag}")
            self._current_pipeline = pipeline_tag
            self._start_time = time.time()

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        if self._current_pipeline is not None:
            elapsed = time.time() - self._start_time
            _banner(f"DONE  {self._current_pipeline}  ({elapsed:.1f} s)")
            self._current_pipeline = None
            self._start_time = None
