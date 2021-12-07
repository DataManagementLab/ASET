from typing import Any, Type

from aset.data.annotations import BaseAnnotation, CurrentMatchIndexAnnotation, SentenceStartCharsAnnotation


def _test_annotation(annotation_class: Type["BaseAnnotation"], value_1: Any, value_2: Any, rep_value: Any) -> None:
    annotation_a: BaseAnnotation = annotation_class(value_1)
    annotation_b: BaseAnnotation = annotation_class(value_2)
    annotation_c: BaseAnnotation = annotation_class(value_2)

    # test __eq__
    assert annotation_a == annotation_a
    assert annotation_a != annotation_b
    assert annotation_b != annotation_a
    assert annotation_b == annotation_c
    assert annotation_c == annotation_b
    assert annotation_a != object()
    assert object() != annotation_a

    # test __str__, __hash__, and __repr__
    assert str(annotation_a) == str(value_1)
    assert repr(annotation_a) == f"{annotation_class.__name__}({repr(value_1)})"
    assert hash(annotation_a) == hash(annotation_a.annotation_str)

    # test value
    assert annotation_a.value == value_1
    assert annotation_b.value == value_2

    annotation_a.value = rep_value
    assert annotation_a.value == rep_value

    # test serialization and deserialization
    annotation_a_deserialized: BaseAnnotation = BaseAnnotation.from_serializable(
        annotation_a.to_serializable(),
        annotation_class.annotation_str
    )
    assert annotation_a_deserialized == annotation_a
    assert annotation_a == annotation_a_deserialized


def test_base_annotation() -> None:
    class Kind1Annotation(BaseAnnotation):
        annotation_str: str = "Kind1Annotation"
        do_serialize: bool = True

        @property
        def value(self) -> int:
            return self._value

        @value.setter
        def value(self, value: int) -> None:
            self._value = value

        def to_serializable(self) -> int:
            return self._value

        @classmethod
        def from_serializable(cls, serialized_annotation: int, annotation_str: str) -> "Kind1Annotation":
            return cls(serialized_annotation)

    class Kind2Annotation(BaseAnnotation):
        annotation_str: str = "Kind2Annotation"
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
        def from_serializable(cls, serialized_annotation: int, annotation_str: str) -> "Kind2Annotation":
            return cls(serialized_annotation)

    kind1_annotation_a: Kind1Annotation = Kind1Annotation(1)
    kind1_annotation_b: Kind1Annotation = Kind1Annotation(2)
    kind1_annotation_c: Kind1Annotation = Kind1Annotation(2)

    kind2_annotation_a: Kind2Annotation = Kind2Annotation(1)
    kind2_annotation_b: Kind2Annotation = Kind2Annotation(2)

    # test __eq__
    assert kind1_annotation_a == kind1_annotation_a
    assert kind1_annotation_b == kind1_annotation_c
    assert kind1_annotation_c == kind1_annotation_b
    assert kind1_annotation_a != kind1_annotation_b
    assert kind1_annotation_b != kind1_annotation_a
    assert kind1_annotation_a != object()
    assert object() != kind1_annotation_a
    assert kind1_annotation_a != kind2_annotation_a

    # test __str__ and __repr__
    assert str(kind1_annotation_a) == "1"
    assert str(kind2_annotation_a) == "1"
    assert repr(kind1_annotation_a) == "Kind1Annotation(1)"
    assert repr(kind2_annotation_b) == "Kind2Annotation(2)"

    # test value
    assert kind1_annotation_a.value == 1
    assert kind1_annotation_b.value == 2

    kind1_annotation_a.value = 3
    assert kind1_annotation_a.value == 3

    # test serialization and deserialization
    kind1_annotation_a_serialized: int = kind1_annotation_a.to_serializable()
    kind1_annotation_a_deserialized: BaseAnnotation = Kind1Annotation.from_serializable(
        kind1_annotation_a_serialized,
        Kind1Annotation.annotation_str
    )
    assert kind1_annotation_a_deserialized == kind1_annotation_a
    assert kind1_annotation_a == kind1_annotation_a_deserialized


def test_sentence_start_chars_annotation() -> None:
    _test_annotation(SentenceStartCharsAnnotation, [1, 2, 3], [3, 10], [0])


def test_current_match_index_annotation() -> None:
    _test_annotation(CurrentMatchIndexAnnotation, 1, 2, 3)
