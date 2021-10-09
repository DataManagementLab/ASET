import abc
import logging
from typing import Optional, Callable, Dict, Any, Type

from aset.config import ConfigurableElement
from aset.data.data import ASETDocumentBase

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

    @abc.abstractmethod
    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: Optional[Callable[[str, float], None]] = None,
            statistics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Derive structured data values for ASETNuggets of the given ASETDocumentBase.

        :param document_base: ASETDocumentBase to work on
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: record to collect statistics
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseNormalizer":
        return NORMALIZERS[config["normalizer_str"]].from_config(config)
