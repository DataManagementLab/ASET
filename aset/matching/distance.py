import abc
import logging
import time
from typing import Union, Dict, Any, Tuple, List, Type

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

from aset.config import ConfigurableElement
from aset.data.data import ASETNugget, ASETAttribute, ASETDocumentBase
from aset.data.signals import LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal, \
    RelativePositionSignal, DistanceCacheIdSignal, POSTagsSignal
from aset.statistics import Statistics
from aset.status import StatusFunction

logger: logging.Logger = logging.getLogger(__name__)

DISTANCES: Dict[str, Type["BaseDistance"]] = {}


def register_distance(distance: Type["BaseDistance"]) -> Type["BaseDistance"]:
    """Register the given distance functions."""
    DISTANCES[distance.distance_str] = distance
    return distance


class BaseDistance(ConfigurableElement, abc.ABC):
    """
    Distance functions compute distances between ASETNuggets and ASETAttributes.

    They are configurable elements. Each distance function comes with an identifier ('distance_str'). The distance
    functions are free in which signals they require as inputs and may also set signal values. They must be able to
    compute distances between pairs of ASETNuggets, pairs of ASETAttributes, or mixed pairs.
    """
    distance_str: str = "BaseDistance"

    def __str__(self) -> str:
        return self.distance_str

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.distance_str == other.distance_str

    def __hash__(self) -> int:
        return hash(self.distance_str)

    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        """
        Execute the distance function on the given document base.

        This method is called before any distances are computed and could be used to prepare the distance calculation.

        :param document_base: ASETDocumentBase to work on
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
        """
        logger.info(f"Execute {self.distance_str}.")
        tick: float = time.time()
        status_fn(f"Running {self.distance_str}...", -1)

        statistics["distance_str"] = self.distance_str
        pass  # default: do nothing

        status_fn(f"Running {self.distance_str}...", 1)
        tack: float = time.time()
        logger.info(f"Executed {self.distance_str} in {tack - tick} seconds.")

    @abc.abstractmethod
    def compute_distance(
            self,
            x: Union[ASETNugget, ASETAttribute],
            y: Union[ASETNugget, ASETAttribute],
            statistics: Statistics
    ) -> float:
        """
        Compute distance between the two given ASETNuggets/ASETAttributes.

        :param x: first ASETNugget/ASETAttribute
        :param y: second ASETNugget/ASETAttribute
        :param statistics: statistics object to collect statistics
        :return: computed distance
        """
        raise NotImplementedError

    def compute_distances(
            self,
            xs: List[Union[ASETNugget, ASETAttribute]],
            ys: List[Union[ASETNugget, ASETAttribute]],
            statistics: Statistics
    ) -> np.array:
        """
        Compute distances between all pairs from two lists of ASETNuggets/ASETAttributes.

        This method exists to speed up the calculation using batching. The default implementation works by calling the
        'compute_distance' method.

        :param xs: first list of ASETNuggets/ASETAttributes
        :param ys: second list of ASETNuggets/ASETAttributes
        :param statistics: statistics object to collect statistics
        :return: matrix of computed distances (row corresponds to xs, column corresponds to ys)
        """
        statistics["num_multi_call"] += 1

        assert xs != [] and ys != [], "Cannot compute distances for an empty list!"

        res: np.array = np.zeros((len(xs), len(ys)))
        for x_ix, x in enumerate(xs):
            for y_ix, y in enumerate(ys):
                res[x_ix, y_ix] = self.compute_distance(x, y, statistics)
        return res

    def feedback_match(
            self,
            x: Union[ASETNugget, ASETAttribute],
            y: Union[ASETNugget, ASETAttribute],
            statistics: Statistics
    ) -> None:
        """
        Give feedback to the distance function that the given pair is a match.

        :param x: first ASETNugget/ASETAttribute
        :param y: second ASETNugget/ASETAttribute
        :param statistics: statistics object to collect statistics
        """
        pass  # default behavior: do nothing

    def feedback_no_match(
            self,
            x: Union[ASETNugget, ASETAttribute],
            y: Union[ASETNugget, ASETAttribute],
            statistics: Statistics
    ) -> None:
        """
        Give feedback to the distance function that the given pair is not a match.

        :param x: first ASETNugget/ASETAttribute
        :param y: second ASETNugget/ASETAttribute
        :param statistics: statistics object to collect statistics
        """
        pass  # default behavior: do nothing

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseDistance":
        return DISTANCES[config["distance_str"]].from_config(config)


@register_distance
class SignalsMeanDistance(BaseDistance):
    """
    Compute the distance as the mean of the distances between the available signals.

    works with signals: LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal,
        RelativePositionSignal, POSTagsSignal
    """
    distance_str: str = "SignalsMeanDistance"

    def __init__(self, signal_strings: List[str]) -> None:
        """
        Initialize the SignalsMeanDistance.

        :param signal_strings: identifiers of the signals to include
        """
        super(SignalsMeanDistance, self).__init__()
        self._signal_strings: List[str] = signal_strings
        logger.debug(f"Initialized distance '{self.distance_str}'.")

    def compute_distance(
            self,
            x: Union[ASETNugget, ASETAttribute],
            y: Union[ASETNugget, ASETAttribute],
            statistics: Statistics
    ) -> float:
        statistics["num_calls"] += 1

        distances: np.array = np.zeros(5)
        is_present: np.array = np.zeros(5)

        label_embedding_signal_str: str = LabelEmbeddingSignal.signal_str
        if label_embedding_signal_str in self._signal_strings and label_embedding_signal_str in x.signals.keys() and \
                label_embedding_signal_str in y.signals.keys():
            cosine_distance: float = cosine(
                x[label_embedding_signal_str],
                y[label_embedding_signal_str]
            )
            distances[0] = min(abs(cosine_distance), 1)
            is_present[0] = 1

        text_embedding_signal_str: str = TextEmbeddingSignal.signal_str
        if text_embedding_signal_str in self._signal_strings and text_embedding_signal_str in x.signals.keys() and \
                text_embedding_signal_str in y.signals.keys():
            cosine_distance: float = cosine(
                x[text_embedding_signal_str],
                y[text_embedding_signal_str]
            )
            distances[1] = min(abs(cosine_distance), 1)
            is_present[1] = 1

        context_sentence_embedding_signal_str: str = ContextSentenceEmbeddingSignal.signal_str
        if context_sentence_embedding_signal_str in self._signal_strings and \
                context_sentence_embedding_signal_str in x.signals.keys() and \
                context_sentence_embedding_signal_str in y.signals.keys():
            cosine_distance: float = cosine(
                x[context_sentence_embedding_signal_str],
                y[context_sentence_embedding_signal_str]
            )
            distances[2] = min(abs(cosine_distance), 1)
            is_present[2] = 1

        relative_position_signal_str: str = RelativePositionSignal.signal_str
        if relative_position_signal_str in self._signal_strings and \
                relative_position_signal_str in x.signals.keys() and relative_position_signal_str in y.signals.keys():
            relative_distance: float = x[relative_position_signal_str] - y[relative_position_signal_str]
            distances[3] = min(abs(relative_distance), 1)
            is_present[3] = 1

        pos_tags_signal_str: str = POSTagsSignal.signal_str
        if pos_tags_signal_str in x.signals.keys() and pos_tags_signal_str in y.signals.keys():
            if x[pos_tags_signal_str] == y[context_sentence_embedding_signal_str]:
                distances[4] = 0
            else:
                distances[4] = 1  # TODO: magic float, measure "distance"
            is_present[4] = 1

        return 1 if np.sum(is_present) == 0 else np.sum(distances) / np.sum(is_present)

    def compute_distances(
            self,
            xs: List[Union[ASETNugget, ASETAttribute]],
            ys: List[Union[ASETNugget, ASETAttribute]],
            statistics: Statistics
    ) -> np.array:
        statistics["num_multi_calls"] += 1

        assert xs != [] and ys != [], "Cannot compute distances for an empty list!"

        signal_strings: List[str] = [
            LabelEmbeddingSignal.signal_str,
            TextEmbeddingSignal.signal_str,
            ContextSentenceEmbeddingSignal.signal_str,
            RelativePositionSignal.signal_str,
            POSTagsSignal.signal_str
        ]

        # check that all xs and all ys contain the same signals
        xs_is_present: np.array = np.zeros(5)
        for idx in range(5):
            if signal_strings[idx] in self._signal_strings and signal_strings[idx] in xs[0].signals.keys():
                xs_is_present[idx] = 1
        for x in xs:
            for idx in range(5):
                if signal_strings[idx] in self._signal_strings:
                    if xs_is_present[idx] == 1 and signal_strings[idx] not in x.signals.keys() or \
                            xs_is_present[idx] == 0 and signal_strings[idx] in x.signals.keys():
                        assert False, "All xs must have the same signals!"

        ys_is_present: np.array = np.zeros(5)
        for idx in range(5):
            if signal_strings[idx] in self._signal_strings and signal_strings[idx] in ys[0].signals.keys():
                ys_is_present[idx] = 1
        for y in ys:
            for idx in range(5):
                if signal_strings[idx] in self._signal_strings:
                    if ys_is_present[idx] == 1 and signal_strings[idx] not in y.signals.keys() or \
                            ys_is_present[idx] == 0 and signal_strings[idx] in y.signals.keys():
                        assert False, "All ys must have the same signals!"

        # compute distances signal by signal
        distances: np.array = np.zeros((len(xs), len(ys)))
        for idx in range(3):
            if xs_is_present[idx] == 1 and ys_is_present[idx] == 1:
                x_embeddings: np.array = np.array([x[signal_strings[idx]] for x in xs])
                y_embeddings: np.array = np.array([y[signal_strings[idx]] for y in ys])
                tmp: np.array = cosine_distances(x_embeddings, y_embeddings)
                distances = np.add(distances, tmp)

        if xs_is_present[3] == 1 and ys_is_present[3] == 1:
            x_positions: np.array = np.array([x[signal_strings[3]] for x in xs])
            y_positions: np.array = np.array([y[signal_strings[3]] for y in ys])
            tmp: np.array = np.zeros((len(x_positions), len(y_positions)))
            for x_ix, x_value in enumerate(x_positions):
                for y_ix, y_value in enumerate(y_positions):
                    tmp[x_ix, y_ix] = np.abs(x_value - y_value)
            distances = np.add(distances, tmp)

        if xs_is_present[4] == 1 and ys_is_present[4] == 1:
            x_values: List[List[str]] = [x[signal_strings[4]] for x in xs]
            y_values: List[List[str]] = [y[signal_strings[4]] for y in ys]
            tmp: np.array = np.ones((len(x_values), len(y_values)))
            for x_ix, x_value in enumerate(x_values):
                for y_ix, y_value in enumerate(y_values):
                    if x_value == y_value:
                        tmp[x_ix, y_ix] = 0
            distances = np.add(distances, tmp)

        actually_present: np.array = xs_is_present * ys_is_present
        if np.sum(actually_present) == 0:
            return np.ones_like(distances)
        else:
            return np.divide(distances, np.sum(actually_present))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SignalsMeanDistance":
        return cls(config["signal_strings"])

    def to_config(self) -> Dict[str, Any]:
        return {
            "distance_str": self.distance_str,
            "signal_strings": self._signal_strings
        }


@register_distance
class CachedDistance(BaseDistance):
    """
    Wrapper for any other distance function that caches its values.

    It will cache values calculated using 'compute_distance' and 'compute_distances', but only resort to the cache when
    'compute_distance' is called.

    works with signals: DistanceCacheIdSignal
    """
    distance_str: str = "CachedDistance"

    def __init__(self, distance: BaseDistance) -> None:
        """
        Initialize the CachedDistance.

        :param distance: actual distance function
        """
        super(CachedDistance, self).__init__()
        self._distance: BaseDistance = distance

        self._distance_cache: Dict[Tuple[int, int], float] = {}
        self._next_cache_id: int = 0
        logger.debug(f"Initialized distance '{self.distance_str}'.")

    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        logger.info(f"Execute {self.distance_str}.")
        tick: float = time.time()
        status_fn(f"Running {self.distance_str}...", -1)

        statistics["distance_str"] = self.distance_str

        for nugget in document_base.nuggets:
            nugget[DistanceCacheIdSignal] = DistanceCacheIdSignal(-1)

        for attribute in document_base.attributes:
            attribute[DistanceCacheIdSignal] = DistanceCacheIdSignal(-1)

        self._distance(document_base, status_fn, statistics)

        status_fn(f"Running {self.distance_str}...", 1)
        tack: float = time.time()
        logger.info(f"Executed {self.distance_str} in {tack - tick} seconds.")

    def compute_distance(
            self,
            x: Union[ASETNugget, ASETAttribute],
            y: Union[ASETNugget, ASETAttribute],
            statistics: Statistics
    ) -> float:
        statistics["num_calls"] += 1

        # attempt to find the distance in the cache
        x_cache_id: int = x[DistanceCacheIdSignal]
        y_cache_id: int = y[DistanceCacheIdSignal]

        if x_cache_id != -1 and y_cache_id != -1:
            if (x_cache_id, y_cache_id) in self._distance_cache.keys():
                statistics["cache_hits"] += 1
                return self._distance_cache[(x_cache_id, y_cache_id)]
            if (y_cache_id, x_cache_id) in self._distance_cache.keys():
                statistics["cache_hits"] += 1
                return self._distance_cache[(y_cache_id, x_cache_id)]

        statistics["cache_misses"] += 1

        # compute the distance
        distance: float = self._distance.compute_distance(x, y, statistics)

        # cache the distance
        if x[DistanceCacheIdSignal] == -1:
            x[DistanceCacheIdSignal] = self._next_cache_id
            self._next_cache_id += 1

        if y[DistanceCacheIdSignal] == -1:
            y[DistanceCacheIdSignal] = self._next_cache_id
            self._next_cache_id += 1

        cache_dict_id: Tuple[int, int] = (x[DistanceCacheIdSignal], y[DistanceCacheIdSignal])
        self._distance_cache[cache_dict_id] = distance

        return distance

    def compute_distances(
            self,
            xs: List[Union[ASETNugget, ASETAttribute]],
            ys: List[Union[ASETNugget, ASETAttribute]],
            statistics: Statistics
    ) -> np.array:
        # compute the distances
        distances: np.array = self._distance.compute_distances(xs, ys, statistics)

        # cache the distances
        for x_ix, x in enumerate(xs):
            for y_ix, y in enumerate(ys):
                if x[DistanceCacheIdSignal] == -1:
                    x[DistanceCacheIdSignal] = self._next_cache_id
                    self._next_cache_id += 1

                if y[DistanceCacheIdSignal] == -1:
                    y[DistanceCacheIdSignal] = self._next_cache_id
                    self._next_cache_id += 1

                cache_dict_id: Tuple[int, int] = (
                    x[DistanceCacheIdSignal],
                    y[DistanceCacheIdSignal]
                )
                self._distance_cache[cache_dict_id] = distances[x_ix, y_ix]

        return distances

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CachedDistance":
        return cls(BaseDistance.from_config(config["distance"]))

    def to_config(self) -> Dict[str, Any]:
        return {
            "distance_str": self.distance_str,
            "distance": self._distance.to_config()
        }
