import abc
import io
import logging
from typing import Any, Dict, List, Type

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

SIGNALS: Dict[str, Type["BaseSignal"]] = {}


def register_signal(signal: Type["BaseSignal"]) -> Type["BaseSignal"]:
    """Register the given signal class."""
    SIGNALS[signal.signal_str] = signal
    return signal


class BaseSignal(abc.ABC):
    """
    Signals for ASETNuggets and ASETAttributes.

    Signals are values associated with ASETNuggets and ASETAttributes. They are used to store properties of the nuggets
    or attributes and could be generated by extractors, normalizers, embedders, etc. Each signal kind comes with an
    identifier ('signal_str') and specifies whether it should be serialized with the nugget/attribute ('do_serialize')
    and how it should be serialized and deserialized ('to_serializable' and 'from_serializable').
    """

    signal_str: str = "BaseSignal"
    do_serialize: bool = False

    def __init__(self, value: Any) -> None:
        """
        Initialize the ASETSignal.

        :param value: value of the signal
        """
        super(BaseSignal, self).__init__()
        self._value: Any = value

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._value)})"

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self._value == other._value

    def __hash__(self) -> int:
        return hash(self.signal_str)

    @property
    def value(self) -> Any:
        """Value of the signal."""
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        """Set the value of the signal."""
        self._value = value

    @abc.abstractmethod
    def to_serializable(self) -> Any:
        """
        Convert the signal to a BSON-serializable representation.

        :return: BSON-serializable representation of the signal
        """
        raise NotImplementedError

    @classmethod
    def from_serializable(cls, serialized_signal: Any, signal_str: str) -> "BaseSignal":
        """
        Create a signal from the BSON-serializable representation.

        :param serialized_signal: BSON-serializable representation of the signal
        :param signal_str: identifier of the signal kind
        :return: deserialized signal
        """
        return SIGNALS[signal_str].from_serializable(serialized_signal, signal_str)


@register_signal
class DistanceCacheIdSignal(BaseSignal):
    """Cache id of the nugget or attribute for cached distance calculators."""

    signal_str: str = "DistanceCacheIdSignal"
    do_serialize: bool = False

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = value

    def to_serializable(self) -> int:
        return self.value

    @classmethod
    def from_serializable(cls, serialized_signal: int, signal_str: str) -> "DistanceCacheIdSignal":
        return cls(serialized_signal)


@register_signal
class CachedDistanceSignal(BaseSignal):
    """Cached distance of the nugget or attribute."""

    signal_str: str = "CachedDistanceSignal"
    do_serialize: bool = False

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value

    def to_serializable(self) -> float:
        return self.value

    @classmethod
    def from_serializable(cls, serialized_signal: float, signal_str: str) -> "CachedDistanceSignal":
        return cls(serialized_signal)


@register_signal
class LabelSignal(BaseSignal):
    """Label of the nugget determined by extractors."""

    signal_str: str = "LabelSignal"
    do_serialize: bool = True

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value: str) -> None:
        self._value = value

    def to_serializable(self) -> str:
        return self.value

    @classmethod
    def from_serializable(cls, serialized_signal: str, signal_str: str) -> "LabelSignal":
        return cls(serialized_signal)


@register_signal
class TreePredecessorSignal(BaseSignal):
    """Parent of an ASETNugget in a tree of ASETNuggets."""

    signal_str: str = "TreePredecessorSignal"
    do_serialize: bool = False

    @property
    def value(self) -> Any:  # no type hint to prevent circular imports
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        self._value = value

    def to_serializable(self) -> Any:
        return self.value

    @classmethod
    def from_serializable(cls, serialized_signal: Any, signal_str: str) -> "TreePredecessorSignal":
        return cls(serialized_signal)


@register_signal
class CachedContextSentenceSignal(BaseSignal):
    """Context sentence and position in context for caching."""

    signal_str: str = "CachedContextSentenceSignal"
    do_serialize: bool = False

    @property
    def value(self) -> Dict[str, Any]:
        return self._value

    @value.setter
    def value(self, value: Dict[str, Any]) -> None:
        self._value = value

    def to_serializable(self) -> Dict[str, Any]:
        return self.value

    @classmethod
    def from_serializable(cls, serialized_signal: Dict[str, Any], signal_str: str) -> "CachedContextSentenceSignal":
        return cls(serialized_signal)


@register_signal
class POSTagsSignal(BaseSignal):
    """POS tags of the nugget's words as determined by extractors."""

    signal_str: str = "POSTagsSignal"
    do_serialize: bool = True

    @property
    def value(self) -> List[str]:
        return self._value

    @value.setter
    def value(self, value: List[str]) -> None:
        self._value = value

    def to_serializable(self) -> List[str]:
        return self.value

    @classmethod
    def from_serializable(cls, serialized_signal: List[str], signal_str: str) -> "POSTagsSignal":
        return cls(serialized_signal)


@register_signal
class UserProvidedExamplesSignal(BaseSignal):
    """User-provided example values/texts for an attribute."""

    signal_str: str = "UserProvidedExamplesSignal"
    do_serialize: bool = True

    @property
    def value(self) -> List[str]:
        return self._value

    @value.setter
    def value(self, value: List[str]) -> None:
        self._value = value

    def to_serializable(self) -> List[str]:
        return self.value

    @classmethod
    def from_serializable(cls, serialized_signal: List[str], signal_str: str) -> "UserProvidedExamplesSignal":
        return cls(serialized_signal)


@register_signal
class RelativePositionSignal(BaseSignal):
    """Relative position of the nugget based on the total length of the document."""

    signal_str: str = "RelativePositionSignal"
    do_serialize: bool = True

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value

    def to_serializable(self) -> float:
        return self.value

    @classmethod
    def from_serializable(cls, serialized_signal: float, signal_str: str) -> "RelativePositionSignal":
        return cls(serialized_signal)


class BaseNumpyArraySignal(BaseSignal, abc.ABC):
    """Base class for signals that have a numpy array as a value."""

    signal_str: str = "BaseNumpyArraySignal"
    do_serialize: bool = True

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and np.array_equal(self._value, other._value)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self._value = value

    def to_serializable(self) -> bytes:
        save_bytes: io.BytesIO = io.BytesIO()
        # noinspection PyTypeChecker
        np.save(save_bytes, self._value, allow_pickle=True)
        return save_bytes.getvalue()

    @classmethod
    def from_serializable(cls, serialized_signal: bytes, signal_str: str) -> "BaseNumpyArraySignal":
        load_bytes: io.BytesIO = io.BytesIO(serialized_signal)
        # noinspection PyTypeChecker
        return cls(np.load(load_bytes, allow_pickle=True))


@register_signal
class LabelEmbeddingSignal(BaseNumpyArraySignal):
    """Embedding of the nugget's label or attribute's name."""

    signal_str: str = "LabelEmbeddingSignal"


@register_signal
class TextEmbeddingSignal(BaseNumpyArraySignal):
    """Embedding of the nugget's text."""

    signal_str: str = "TextEmbeddingSignal"


@register_signal
class ContextSentenceEmbeddingSignal(BaseNumpyArraySignal):
    """Embedding of the nugget's textual context sentence."""

    signal_str: str = "ContextSentenceEmbeddingSignal"
