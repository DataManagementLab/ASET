import abc
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ConfigurableElement(abc.ABC):
    """
    Base class for all configurable elements.

    A configurable element is a class (e.g. extractors, normalizers, embedders) that can be configured. The element's
    configuration must be serializable ('to_config'), and the exact same element must be reproducible from its
    serialized configuration ('from_config').
    """

    @abc.abstractmethod
    def to_config(self) -> Dict[str, Any]:
        """
        Obtain a JSON-serializable representation of the element.

        :return: JSON-serializable representation of the element
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConfigurableElement":
        """
        Create the element from its JSON-serializable representation.

        :param config: JSON-serializable representation of the element
        :return: element created from the JSON-serializable representation
        """
        raise NotImplementedError
