import abc
import logging
from typing import Any, Dict, List, Type

logger: logging.Logger = logging.getLogger(__name__)

ANNOTATIONS: Dict[str, Type["BaseAnnotation"]] = {}


def register_annotation(annotation: Type["BaseAnnotation"]) -> Type["BaseAnnotation"]:
    """Register the given annotation class."""
    ANNOTATIONS[annotation.annotation_str] = annotation
    return annotation


class BaseAnnotation(abc.ABC):
    """
    Annotations for ASETDocuments.

    Annotations are values associated with ASETDocuments. They are used to store properties of the documents and could
    be generated by extractors, normalizers, etc. Each annotation kind comes with an identifier ('annotation_str') and
    specifies whether it should be serialized with the document ('do_serialize') and how it should be serialized and
    deserialized ('to_serializable' and 'from_serializable').
    """

    annotation_str: str = "BaseAnnotation"
    do_serialize: bool = False

    def __init__(self, value: Any) -> None:
        """
        Initialize the ASETAnnotation.

        :param value: value of the annotation
        """
        super(BaseAnnotation, self).__init__()
        self._value: Any = value

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._value)})"

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self._value == other._value

    def __hash__(self) -> int:
        return hash(self.annotation_str)

    @property
    def value(self) -> Any:
        """Value of the annotation."""
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        """Set the value of the annotation."""
        self._value = value

    @abc.abstractmethod
    def to_serializable(self) -> Any:
        """
        Convert the annotation to a BSON-serializable representation.

        :return: BSON-serializable representation of the annotation
        """
        raise NotImplementedError

    @classmethod
    def from_serializable(cls, serialized_annotation: Any, annotation_str: str) -> "BaseAnnotation":
        """
        Create an annotation from the BSON-serializable representation.

        :param serialized_annotation: BSON-serializable representation of the annotation
        :param annotation_str: identifier of the annotation kind
        :return: deserialized annotation
        """
        return ANNOTATIONS[annotation_str].from_serializable(serialized_annotation, annotation_str)


@register_annotation
class SentenceStartCharsAnnotation(BaseAnnotation):
    """Sentence boundaries as a list of indices of the first characters in each sentence."""

    annotation_str: str = "SentenceStartCharsAnnotation"
    do_serialize: bool = True

    @property
    def value(self) -> List[int]:
        return self._value

    @value.setter
    def value(self, value: List[int]) -> None:
        self._value = value

    def to_serializable(self) -> List[int]:
        return self._value

    @classmethod
    def from_serializable(cls, serialized_annotation: List[int], annotation_str: str) -> "SentenceStartCharsAnnotation":
        return cls(serialized_annotation)


@register_annotation
class CurrentMatchIndexAnnotation(BaseAnnotation):
    """Index of the nugget that is currently considered as the match."""

    annotation_str: str = "CurrentMatchIndexAnnotation"
    do_serialize: bool = False

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = value

    def to_serializable(self) -> int:
        return self._value

    @classmethod
    def from_serializable(cls, serialized_annotation: int, annotation_str: str) -> "CurrentMatchIndexAnnotation":
        return cls(serialized_annotation)