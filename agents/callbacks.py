"""Module for callbacks for the agents"""
from types import TracebackType
from typing import Self, override

from langchain_core.callbacks import BaseCallbackHandler

from tqdm import tqdm


__all__ = ["BatchBarCallback"]


class BatchBarCallback(BaseCallbackHandler):
    """Callback for support tqdm on Runnable.batch method."""

    def __init__(self, total, **kwargs):
        super().__init__()
        # define a progress bar
        self.count = 0
        self.progress_bar = tqdm(total=total, **kwargs)

    @override
    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        if self.count >= self.progress_bar.total:
            self.count = 0
            self.progress_bar.reset()
        self.count += 1
        self.progress_bar.update(1)
        self.progress_bar.refresh()

    def __enter__(self) -> Self:
        self.progress_bar.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 exc_traceback: TracebackType | None) -> None:
        self.progress_bar.__exit__(exc_type, exc_value, exc_traceback)

    def __del__(self):
        self.progress_bar.__del__()
