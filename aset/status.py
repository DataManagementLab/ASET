import logging
from typing import Optional, Callable

logger: logging.Logger = logging.getLogger(__name__)


class StatusFunction:

    def __init__(self, callback_fn: Optional[Callable[[str, float], None]] = None) -> None:
        super(StatusFunction, self).__init__()
        self._callback_fn: Optional[Callable[[str, float], None]] = callback_fn

    def __call__(self, message: str, progress: float):
        if progress == -1:
            logger.info(f"{message} ~%")
        else:
            logger.info(f"{message} {round(progress * 100)}%")

        if self._callback_fn is not None:
            self._callback_fn(message, progress)
