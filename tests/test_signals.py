from typing import Any, Type

import numpy as np

from aset.data.signals import BaseSignal, DistanceCacheIdSignal, LabelSignal, \
    UserProvidedExamplesSignal, RelativePositionSignal, LabelEmbeddingSignal, TextEmbeddingSignal, \
    ContextSentenceEmbeddingSignal, BaseNumpyArraySignal


def _test_signal(signal_class: Type["BaseSignal"], value_1: Any, value_2: Any, rep_value: Any) -> None:
    signal_a: BaseSignal = signal_class(value_1)
    signal_b: BaseSignal = signal_class(value_2)
    signal_c: BaseSignal = signal_class(value_2)

    # test __eq__
    assert signal_a == signal_a
    assert signal_a != signal_b
    assert signal_b != signal_a
    assert signal_b == signal_c
    assert signal_c == signal_b
    assert signal_a != object()
    assert object() != signal_a

    # test __str__, __hash__, and __repr__
    assert str(signal_a) == str(value_1)
    assert repr(signal_a) == f"{signal_class.__name__}({repr(value_1)})"
    assert hash(signal_a) == hash(signal_a.signal_str)

    # test value
    assert signal_a.value == value_1
    assert signal_b.value == value_2

    signal_a.value = rep_value
    assert signal_a.value == rep_value

    # test serialization and deserialization
    signal_a_deserialized: BaseSignal = BaseSignal.from_serializable(
        signal_a.to_serializable(),
        signal_class.signal_str
    )
    assert signal_a_deserialized == signal_a
    assert signal_a == signal_a_deserialized


def _test_numpy_array_signal(
        signal_class: Type["BaseNumpyArraySignal"],
        value_1: Any,
        value_2: Any,
        rep_value: Any
) -> None:
    signal_a: BaseNumpyArraySignal = signal_class(value_1)
    signal_b: BaseNumpyArraySignal = signal_class(value_2)
    signal_c: BaseNumpyArraySignal = signal_class(value_2)

    # test __eq__
    assert signal_a == signal_a
    assert signal_a != signal_b
    assert signal_b != signal_a
    assert signal_b == signal_c
    assert signal_c == signal_b
    assert signal_a != object()
    assert object() != signal_a

    # test __str__ and __repr__
    assert str(signal_a) == str(value_1)
    assert repr(signal_a) == f"{signal_class.__name__}({repr(value_1)})"

    # test value
    assert all(signal_a.value == value_1)
    assert all(signal_b.value == value_2)

    signal_a.value = rep_value
    assert all(signal_a.value == rep_value)

    # test serialization and deserialization
    signal_a_deserialized: BaseSignal = BaseSignal.from_serializable(
        signal_a.to_serializable(),
        signal_class.signal_str
    )
    assert signal_a_deserialized == signal_a
    assert signal_a == signal_a_deserialized


def test_base_signal() -> None:
    class Kind1Signal(BaseSignal):
        signal_str: str = "Kind1Signal"
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
        def from_serializable(cls, serialized_signal: int, signal_str: str) -> "Kind1Signal":
            return cls(serialized_signal)

    class Kind2Signal(BaseSignal):
        signal_str: str = "Kind2Signal"
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
        def from_serializable(cls, serialized_signal: int, signal_str: str) -> "Kind2Signal":
            return cls(serialized_signal)

    kind1_signal_a: Kind1Signal = Kind1Signal(1)
    kind1_signal_b: Kind1Signal = Kind1Signal(2)
    kind1_signal_c: Kind1Signal = Kind1Signal(2)

    kind2_signal_a: Kind2Signal = Kind2Signal(1)

    # test __eq__
    assert kind1_signal_a == kind1_signal_a
    assert kind1_signal_b == kind1_signal_c
    assert kind1_signal_c == kind1_signal_b
    assert kind1_signal_a != kind1_signal_b
    assert kind1_signal_b != kind1_signal_a
    assert kind1_signal_a != object()
    assert object() != kind1_signal_a
    assert kind1_signal_a != kind2_signal_a

    # test __str__ and __repr__
    assert str(kind1_signal_a) == "1"
    assert str(kind2_signal_a) == "1"
    assert repr(kind1_signal_a) == "Kind1Signal(1)"
    assert repr(kind2_signal_a) == "Kind2Signal(1)"

    # test value
    assert kind1_signal_a.value == 1
    assert kind1_signal_b.value == 2

    kind1_signal_a.value = 3
    assert kind1_signal_a.value == 3

    # test serialization and deserialization
    kind1_signal_a_serialized: int = kind1_signal_a.to_serializable()
    kind1_signal_a_deserialized: BaseSignal = Kind1Signal.from_serializable(
        kind1_signal_a_serialized,
        Kind1Signal.signal_str
    )
    assert kind1_signal_a_deserialized == kind1_signal_a
    assert kind1_signal_a == kind1_signal_a_deserialized


def test_distance_cache_id_signal() -> None:
    _test_signal(DistanceCacheIdSignal, 1, 2, -1)


def test_cached_distance_signal() -> None:
    _test_signal(DistanceCacheIdSignal, 0.7, 0.1, 0.5)


def test_label_signal() -> None:
    _test_signal(LabelSignal, "label-1", "label-2", "label-3")


def test_user_provided_examples_signal() -> None:
    _test_signal(UserProvidedExamplesSignal, ["a", "b"], ["c"], [])


def test_relative_position_signal() -> None:
    _test_signal(RelativePositionSignal, 0.2, 0.6, 0.01)


# noinspection PyTypeChecker
def test_label_embedding_signal() -> None:
    _test_numpy_array_signal(
        LabelEmbeddingSignal,
        np.array([0.2, 0.3, 0.4]),
        np.array([0.02, 0.1, 2.2]),
        np.array([0.1])
    )


# noinspection PyTypeChecker
def test_text_embedding_signal() -> None:
    _test_numpy_array_signal(
        TextEmbeddingSignal,
        np.array([0.2, 0.3, 0.4]),
        np.array([0.02, 0.1, 2.2]),
        np.array([0.1])
    )


# noinspection PyTypeChecker
def test_context_sentence_embedding_signal() -> None:
    _test_numpy_array_signal(
        ContextSentenceEmbeddingSignal,
        np.array([0.2, 0.3, 0.4]),
        np.array([0.02, 0.1, 2.2]),
        np.array([0.1])
    )
