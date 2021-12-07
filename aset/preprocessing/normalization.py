import abc
import logging
from typing import Any, Dict, Type

from aset.config import ConfigurableElement
from aset.data.data import ASETDocumentBase
from aset.statistics import Statistics
from aset.status import StatusFunction

logger: logging.Logger = logging.getLogger(__name__)

NORMALIZERS: Dict[str, Type["BaseNormalizer"]] = {}


def register_normalizer(normalizer: Type["BaseNormalizer"]) -> Type["BaseNormalizer"]:
    """Register the given normalizer class."""
    NORMALIZERS[normalizer.normalizer_str] = normalizer
    return normalizer


class BaseNormalizer(ConfigurableElement, abc.ABC):
    """
    Normalizers work on ASETNuggets to derive their structured data values.

    They are configurable elements and should be applied in the preprocessing phase. A normalizer does not have to work
    with every nugget in the document base. Each normalizer comes with an identifier ('normalizer_str').
    """

    normalizer_str: str = "BaseNormalizer"

    def __str__(self) -> str:
        return self.normalizer_str

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.normalizer_str == other.normalizer_str

    def __hash__(self) -> int:
        return hash(self.normalizer_str)

    def _use_status_fn(self, status_fn: StatusFunction, document_base: ASETDocumentBase, ix: int) -> None:
        """
        Helper method that calls the status function at regular intervals.

        :param status_fn: status function to call
        :param document_base: ASETDocumentBase to work on
        :param ix: index of the current document
        """
        if ix == 0:
            status_fn(f"Running {self.normalizer_str}...", 0)
        else:
            num_documents: int = len(document_base.documents)
            interval: int = num_documents // 10
            if interval != 0 and ix % interval == 0:
                status_fn(f"Running {self.normalizer_str}...", ix / num_documents)

    @abc.abstractmethod
    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        """
        Derive structured data values for ASETNuggets of the given ASETDocumentBase.

        :param document_base: ASETDocumentBase to work on
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseNormalizer":
        return NORMALIZERS[config["normalizer_str"]].from_config(config)
