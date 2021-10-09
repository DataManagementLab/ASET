import abc
import logging
import random
import time
from typing import Dict, Any, Optional, Callable, List, Type

import numpy as np
from scipy.special import softmax

from aset.config import ConfigurableElement
from aset.data.annotations import SentenceStartCharsAnnotation, CurrentMatchIndexAnnotation
from aset.data.data import ASETDocumentBase, ASETNugget, ASETAttribute, ASETDocument
from aset.data.signals import CachedDistanceSignal, UserProvidedExamplesSignal, CachedContextSentenceSignal
from aset.matching.distance import BaseDistance
from aset.preprocessing.embedding import BaseEmbedder

logger: logging.Logger = logging.getLogger(__name__)

MATCHING_PHASES: Dict[str, Type["BaseMatchingPhase"]] = {}


def register_matching_phase(matching_phase: Type["BaseMatchingPhase"]) -> Type["BaseMatchingPhase"]:
    """Register the given matching phase."""
    MATCHING_PHASES[matching_phase.matching_phase_str] = matching_phase
    return matching_phase


class BaseMatchingPhase(ConfigurableElement, abc.ABC):
    """
    Matching phase that attempts to find matching ASETNuggets for the ASETAttributes.

    This is the base class for all matching phases. Each matching phase is a configurable element. Different matching
    phases may have different '__call__' method signatures.
    """
    matching_phase_str: str = "BaseMatchingPhase"

    def __str__(self) -> str:
        return self.matching_phase_str

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.matching_phase_str == other.matching_phase_str

    @abc.abstractmethod
    def __call__(
            self,
            document_base: ASETDocumentBase,
            feedback_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
            status_fn: Optional[Callable[[str, float], None]] = None,
            statistics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Find matching ASETNuggets for the ASETAttributes.

        :param document_base: document base to work on
        :param feedback_fn: callback function for feedback
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: record to collect statistics
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseMatchingPhase":
        return MATCHING_PHASES[config["matching_phase_str"]].from_config(config)


@register_matching_phase
class TreeSearchMatchingPhase(BaseMatchingPhase):
    """
    Matching phase that performs a tree-search to find matching nuggets.

    The user can provide example values for each attribute. Furthermore, he gives yes/no feedback on potential matches.

    works with signals: UserProvidedExamplesSignal, CachedDistanceSignal
    """
    matching_phase_str: str = "TreeSearchMatchingPhase"

    def __init__(
            self,
            distance: BaseDistance,
            examples_embedder: BaseEmbedder,
            max_num_feedback: int,
            max_children: int,
            max_distance: float,
            exploration_factor: float
    ) -> None:
        """
        Initialize the TreeSearchMatchingPhase.

        :param distance: distance function
        :param examples_embedder: embedder for the user-provided examples
        :param max_num_feedback: maximum number of user interactions per attribute
        :param max_children: maximum number of child nodes in the search tree
        :param max_distance: maximum distance at which nuggets will be accepted
        :param exploration_factor: how far away the samples presented to the user should be from confirmed matches
        """
        super(TreeSearchMatchingPhase, self).__init__()
        self._distance: BaseDistance = distance
        self._examples_embedder: BaseEmbedder = examples_embedder
        self._max_num_feedback: int = max_num_feedback
        self._max_children: int = max_children
        self._max_distance: float = max_distance
        self._exploration_factor: float = exploration_factor
        logger.debug(f"Initialized matching phase '{self.matching_phase_str}'.")

    def __call__(  # TODO: update feedback_fn and remove examples_fn
            self,
            document_base: ASETDocumentBase,
            feedback_fn: Callable[[ASETNugget, ASETAttribute], bool],
            examples_fn: Optional[Callable[[ASETAttribute], List[str]]] = None,
            status_fn: Optional[Callable[[str, float], None]] = None,
            statistics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Find matching ASETNuggets for the ASETAttributes.

        :param document_base: document base to work on
        :param feedback_fn: callback function for yes/no feedback on potential matches
        :param examples_fn: callback function to incorporate user-provided examples
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: record to collect statistics
        """
        logger.info(f"Execute matching phase '{self.matching_phase_str}' on document base with "
                    f"{len(document_base.documents)} documents and {len(document_base.attributes)} attributes.")
        tick: float = time.time()

        if statistics is not None:
            statistics["matching"] = {}
            statistics: Dict[str, Any] = statistics["matching"]
            statistics["matching_phase_str"] = self.matching_phase_str
            statistics["num_documents"] = len(document_base.documents)
            statistics["num_nuggets"] = len(document_base.nuggets)
            statistics["attributes"] = {}
            statistics["distance"] = {"distance_str": self._distance.distance_str}
            statistics["examples"] = {}
            statistics["examples_embedder"] = {}

        # incorporate examples
        if examples_fn is not None:
            logger.info(f"Incorporate examples on document base with {len(document_base.documents)} and "
                        f"{len(document_base.attributes)} attributes.")
            tik: float = time.time()
            for attribute in document_base.attributes:
                examples: List[str] = examples_fn(attribute)
                if statistics is not None:
                    statistics["examples"][attribute.name] = examples
                attribute.signals[UserProvidedExamplesSignal.signal_str] = UserProvidedExamplesSignal(examples)

            if statistics is not None:
                self._examples_embedder(document_base, status_fn=status_fn, statistics=statistics["examples_embedder"])
            else:
                self._examples_embedder(document_base, status_fn=status_fn)
            tak: float = time.time()
            logger.info(f"Incorporated examples on document base with {len(document_base.documents)} and "
                        f"{len(document_base.attributes)} attributes in {tak - tik} seconds.")

        logger.info(f"Execute distance '{self._distance.distance_str}' on document base with "
                    f"{len(document_base.documents)} and {len(document_base.attributes)} attributes.")
        tik: float = time.time()
        self._distance(document_base, status_fn, statistics["distance"])
        tak: float = time.time()
        logger.info(f"Executed distance '{self._distance.distance_str}' on document base with "
                    f"{len(document_base.documents)} and {len(document_base.attributes)} attributes in "
                    f"{tak - tik} seconds.")

        # match attribute-by-attribute
        for attribute in document_base.attributes:
            logger.info(f"Match attribute '{attribute.name}'.")
            tik: float = time.time()
            if statistics is not None:
                statistics["attributes"][attribute.name] = {}
                statistics["attributes"][attribute.name]["positive_feedback"] = 0
                statistics["attributes"][attribute.name]["negative_feedback"] = 0
                statistics["attributes"][attribute.name]["confirmed_matches"] = 0
                statistics["attributes"][attribute.name]["guessed_matches"] = 0

            num_feedback: int = 0

            remaining_nuggets: List[ASETNugget] = []
            matching_nuggets: List[ASETNugget] = []
            queue: List[ASETNugget] = []

            # initialize the distances with the distances to the attribute
            distances_to_attribute: bool = True
            if document_base.nuggets != []:
                if statistics is not None:
                    distances: np.ndarray = self._distance.compute_distances([attribute], document_base.nuggets,
                                                                             statistics=statistics["distance"])
                else:
                    distances: np.ndarray = self._distance.compute_distances([attribute], document_base.nuggets)
                for nugget, distance in zip(document_base.nuggets, distances[0]):
                    nugget.signals[CachedDistanceSignal.signal_str] = CachedDistanceSignal(distance)
                    remaining_nuggets.append(nugget)

            # for nugget in document_base.nuggets:
            #     distance: float = self._distance.compute_distance(attribute, nugget, statistics["distance"])
            #     nugget.signals[CachedDistanceSignal.signal_str] = CachedDistanceSignal(distance)
            #     remaining_nuggets.append(nugget)

            while num_feedback < self._max_num_feedback:

                # find a root
                root_iteration: int = 1
                while queue == [] and num_feedback < self._max_num_feedback:
                    temperature: float = 0.001 * (root_iteration ** 2)
                    weights: List[float] = [(1 - nugget.signals[CachedDistanceSignal.signal_str].value) / temperature
                                            for nugget in remaining_nuggets]
                    softmax_weights: np.ndarray = softmax(weights)

                    nugget: ASETNugget = random.choices(remaining_nuggets, weights=softmax_weights)[0]

                    num_feedback += 1
                    if feedback_fn(nugget, attribute):
                        if statistics is not None:
                            statistics["attributes"][attribute.name]["positive_feedback"] += 1
                        remaining_nuggets: List[ASETNugget] = [n for n in remaining_nuggets if
                                                               n.document is not nugget.document]
                        matching_nuggets.append(nugget)
                        queue.append(nugget)
                    else:
                        if statistics is not None:
                            statistics["attributes"][attribute.name]["negative_feedback"] += 1
                    root_iteration += 1

                # if the distances are still the distances to the attribute, set them to one
                if distances_to_attribute:
                    for nugget in remaining_nuggets:
                        nugget.signals[CachedDistanceSignal.signal_str].value = 1
                    distances_to_attribute = False

                # explore the tree
                while queue != [] and num_feedback < self._max_num_feedback:
                    nugget = queue.pop(0)  # pop the first element ==> FIFO queue

                    # compute the distance of the current node to all other already expanded nodes
                    distance: float = 1
                    for nug in matching_nuggets:
                        if nug is not nugget:
                            if statistics is not None:
                                dist: float = self._distance.compute_distance(nugget, nug,
                                                                              statistics=statistics["distance"])
                            else:
                                dist: float = self._distance.compute_distance(nugget, nug)
                            if dist < distance:
                                distance: float = dist

                    if len(matching_nuggets) == 1:
                        distance: float = 0

                    # compute the distances to the current, update distances if necessary and find possible samples
                    samples: List[ASETNugget] = []
                    if remaining_nuggets != []:
                        if statistics is not None:
                            new_distances: np.ndarray = self._distance.compute_distances(
                                [nugget],
                                remaining_nuggets,
                                statistics=statistics["distance"]
                            )
                        else:
                            new_distances: np.ndarray = self._distance.compute_distances([nugget], remaining_nuggets)
                        for nug, new_dist in zip(remaining_nuggets, new_distances[0]):
                            # explore only if closer to this one than to any other one
                            if new_dist < nug.signals[CachedDistanceSignal.signal_str].value:
                                nug.signals[CachedDistanceSignal.signal_str].value = new_dist
                                if new_dist / self._exploration_factor > distance:  # explore farther away
                                    samples.append(nug)

                    samples: List[ASETNugget] = sorted(
                        samples,
                        key=lambda x: x.signals[CachedDistanceSignal.signal_str].value
                    )
                    samples: List[ASETNugget] = samples[:self._max_children]

                    # query the user about the samples
                    new_matching: List[ASETNugget] = []
                    for nug in samples:
                        if num_feedback < self._max_num_feedback:
                            num_feedback += 1
                            if feedback_fn(nug, attribute):
                                if statistics is not None:
                                    statistics["attributes"][attribute.name]["positive_feedback"] += 1
                                matching_nuggets.append(nug)
                                new_matching.append(nug)
                            else:
                                if statistics is not None:
                                    statistics["attributes"][attribute.name]["negative_feedback"] += 1

                    # update queue and remaining
                    queue += new_matching
                    for nug in new_matching:
                        remaining_nuggets: List[ASETNugget] = \
                            [n for n in remaining_nuggets if n.document is not nug.document]

            # update the distances with the remaining stack
            for nugget in queue:
                if remaining_nuggets != []:
                    if statistics is not None:
                        new_distances: np.ndarray = self._distance.compute_distances(
                            [nugget],
                            remaining_nuggets,
                            statistics=statistics["distance"]
                        )
                    else:
                        new_distances: np.ndarray = self._distance.compute_distances([nugget], remaining_nuggets)
                    for nug, new_dist in zip(remaining_nuggets, new_distances[0]):
                        if new_dist < nug.signals[CachedDistanceSignal.signal_str].value:
                            nug.signals[CachedDistanceSignal.signal_str].value = new_dist

            # match based on the calculated distances
            for nugget in matching_nuggets:
                if statistics is not None:
                    statistics["attributes"][attribute.name]["confirmed_matches"] += 1
                nugget.document.attribute_mappings[attribute.name] = [nugget]
                nugget.signals[CachedDistanceSignal.signal_str].value = 0

            for nugget in remaining_nuggets:
                distance: float = nugget.signals[CachedDistanceSignal.signal_str].value
                if attribute.name not in nugget.document.attribute_mappings.keys():
                    nugget.document.attribute_mappings[attribute.name] = []
                existing_nuggets: List[ASETNugget] = nugget.document.attribute_mappings[attribute.name]

                if existing_nuggets == [] or \
                        (distance < existing_nuggets[0].signals[CachedDistanceSignal.signal_str].value and
                         distance < self._max_distance):
                    if existing_nuggets == []:
                        if statistics is not None:
                            statistics["attributes"][attribute.name]["guessed_matches"] += 1
                    nugget.document.attribute_mappings[attribute.name] = [nugget]

            tak: float = time.time()
            logger.info(f"Matched attribute '{attribute.name}' in {tak - tik} seconds.")

        tack: float = time.time()
        logger.info(f"Executed matching phase '{self.matching_phase_str}' on document base with "
                    f"{len(document_base.documents)} documents and {len(document_base.attributes)} attributes "
                    f"in {tack - tick} seconds.")

    def to_config(self) -> Dict[str, Any]:
        return {
            "matching_phase_str": self.matching_phase_str,
            "distance": self._distance.to_config(),
            "examples_embedder": self._examples_embedder.to_config(),
            "max_num_feedback": self._max_num_feedback,
            "max_children": self._max_children,
            "max_distance": self._max_distance,
            "exploration_factor": self._exploration_factor
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TreeSearchMatchingPhase":
        distance: BaseDistance = BaseDistance.from_config(config["distance"])
        examples_embedder: BaseEmbedder = BaseEmbedder.from_config(config["examples_embedder"])
        return cls(distance, examples_embedder, config["max_num_feedback"], config["max_children"],
                   config["max_distance"], config["exploration_factor"])


@register_matching_phase
class RankingBasedMatchingPhase(BaseMatchingPhase):
    """
    Matching phase that displays a ranked list of nuggets to the user for feedback.

    works with signals: CachedContextSentenceSignal, CachedDistanceSignal
    """
    matching_phase_str: str = "RankingBasedMatchingPhase"

    def __init__(
            self,
            distance: BaseDistance,
            max_num_feedback: int,
            len_ranked_list: int,
            max_distance: float
    ) -> None:
        """
        Initialize the RankingBasedMatchingPhase.

        :param distance: distance function
        :param max_num_feedback: maximum number of user interactions per attribute
        :param len_ranked_list: length of the ranked list of nuggets presented to the user for feedback
        :param max_distance: maximum distance at which nuggets will be accepted
        """
        super(RankingBasedMatchingPhase, self).__init__()
        self._distance: BaseDistance = distance
        self._max_num_feedback: int = max_num_feedback
        self._len_ranked_list: int = len_ranked_list
        self._max_distance: float = max_distance
        logger.debug(f"Initialized matching phase '{self.matching_phase_str}'.")

    def __call__(
            self,
            document_base: ASETDocumentBase,
            feedback_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
            status_fn: Optional[Callable[[str, float], None]] = None,
            statistics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Find matching ASETNuggets for the ASETAttributes.

        :param document_base: document base to work on
        :param feedback_fn: callback function for feedback
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: record to collect statistics
        """
        logger.info(f"Execute matching phase '{self.matching_phase_str}' on document base with "
                    f"{len(document_base.documents)} documents and {len(document_base.attributes)} attributes.")
        tick: float = time.time()

        if statistics is not None:
            statistics["matching"] = {}
            statistics: Dict[str, Any] = statistics["matching"]
            statistics["matching_phase_str"] = self.matching_phase_str
            statistics["num_documents"] = len(document_base.documents)
            statistics["num_nuggets"] = len(document_base.nuggets)
            statistics["attributes"] = {}
            statistics["distance"] = {"distance_str": self._distance.distance_str}

        # cache the context sentences
        logger.info("Cache the context sentences.")
        tik: float = time.time()

        for nugget in document_base.nuggets:
            sent_start_chars: List[int] = nugget.document.annotations[SentenceStartCharsAnnotation.annotation_str].value
            context_start_char: int = 0
            context_end_char: int = 0
            for ix, sent_start_char in enumerate(sent_start_chars):
                if sent_start_char > nugget.start_char:
                    if ix == 0:
                        context_start_char: int = 0
                        context_end_char: int = sent_start_char
                        break
                    else:
                        context_start_char: int = sent_start_chars[ix - 1]
                        context_end_char: int = sent_start_char
                        break
            else:
                if sent_start_chars != []:
                    context_start_char: int = sent_start_chars[-1]
                    context_end_char: int = len(nugget.document.text)

            context_sentence: str = nugget.document.text[context_start_char:context_end_char]
            start_in_context: int = nugget.start_char - context_start_char
            end_in_context: int = nugget.end_char - context_start_char

            nugget.signals[CachedContextSentenceSignal.signal_str] = CachedContextSentenceSignal({
                "text": context_sentence,
                "start_char": start_in_context,
                "end_char": end_in_context
            })

        tak: float = time.time()
        logger.info(f"Cached context sentences in {tak - tik} seconds.")

        for attribute in document_base.attributes:
            logger.info(f"Matching attribute '{attribute.name}'.")
            remaining_documents: List[ASETDocument] = []

            # compute initial distances as distances to label
            logger.info("Compute initial distances and initialize documents.")
            tik: float = time.time()

            distances = self._distance.compute_distances([attribute], document_base.nuggets)[0]
            for nugget, distance in zip(document_base.nuggets, distances):
                nugget.signals[CachedDistanceSignal.signal_str] = CachedDistanceSignal(distance)

            for document in document_base.documents:
                index: int = -1
                dist: float = 2
                for ix, nugget in enumerate(document.nuggets):
                    if nugget.signals[CachedDistanceSignal.signal_str].value < dist:
                        dist = nugget.signals[CachedDistanceSignal.signal_str].value
                        index = ix
                if index != -1:
                    document.annotations[CurrentMatchIndexAnnotation.annotation_str] = \
                        CurrentMatchIndexAnnotation(index)
                    remaining_documents.append(document)
                else:  # document has no nuggets
                    document.attribute_mappings[attribute.name] = []
                    if statistics is not None:
                        statistics["num_document_with_no_nuggets"] = \
                            statistics.get("num_document_with_no_nuggets", 0) + 1

            tak: float = time.time()
            logger.info(f"Computed initial distances and initialized documents in {tak - tik} seconds.")

            # iterative user interactions
            logger.info("Execute interactive matching.")
            tik: float = time.time()
            num_feedback: int = 0
            continue_matching: bool = True
            while continue_matching and num_feedback < self._max_num_feedback and remaining_documents != []:
                # sort remaining documents by distance
                remaining_documents = list(sorted(
                    remaining_documents,
                    key=lambda x: x.nuggets[x.annotations[CurrentMatchIndexAnnotation.annotation_str].value].signals[
                        CachedDistanceSignal.signal_str].value,
                    reverse=True
                ))

                # present documents to the user for feedback
                feedback_nuggets: List[ASETNugget] = []
                for doc in remaining_documents[:self._len_ranked_list]:
                    feedback_nuggets.append(
                        doc.nuggets[doc.annotations[CurrentMatchIndexAnnotation.annotation_str].value]
                    )
                feedback_result: Dict[str, Any] = feedback_fn({
                    "nuggets": feedback_nuggets,
                    "attribute": attribute
                })

                if feedback_result["message"] == "stop-interactive-matching":
                    if statistics is not None:
                        statistics["stopped_matching_by_hand"] = True
                    continue_matching = False
                elif feedback_result["message"] == "no-match-in-document":
                    if statistics is not None:
                        statistics["num_no_match_in_document"] = statistics.get("num_no_match_in_document", 0) + 1
                    feedback_result["nugget"].document.attribute_mappings[attribute.name] = []
                    remaining_documents.remove(feedback_result["nugget"].document)
                elif feedback_result["message"] == "is-match":
                    confirmed_nugget: ASETNugget = feedback_result["nugget"]
                    if statistics is not None:
                        statistics["num_confirmed_match"] = statistics.get("num_confirmed_match", 0) + 1
                    confirmed_nugget.document.attribute_mappings[attribute.name] = [confirmed_nugget]
                    remaining_documents.remove(feedback_result["nugget"].document)

                    # update the distances for the other documents
                    for document in remaining_documents:
                        new_distances: np.array = \
                        self._distance.compute_distances([confirmed_nugget], document.nuggets)[0]
                        for ix, (nugget, new_distance) in enumerate(zip(document.nuggets, new_distances)):
                            current_guess: ASETNugget = document.nuggets[
                                document.annotations[CurrentMatchIndexAnnotation.annotation_str].value]
                            if new_distance < current_guess.signals[CachedDistanceSignal.signal_str].value:
                                document.annotations[CurrentMatchIndexAnnotation.annotation_str].value = ix
                            if new_distance < nugget.signals[CachedDistanceSignal.signal_str].value:
                                nugget.signals[CachedDistanceSignal.signal_str].value = new_distance
            tak: float = time.time()
            logger.info(f"Executed interactive matching in {tak - tik} seconds.")

            # update remaining documents
            logger.info("Update remaining documents.")
            tik: float = time.time()

            for document in remaining_documents:
                current_guess: ASETNugget = document.nuggets[
                    document.annotations[CurrentMatchIndexAnnotation.annotation_str].value]
                if current_guess.signals[CachedDistanceSignal.signal_str].value < self._max_distance:
                    if statistics is not None:
                        statistics["num_guessed_match"] = statistics.get("num_guessed_match", 0) + 1
                    document.attribute_mappings[attribute.name] = [current_guess]
                else:
                    if statistics is not None:
                        statistics["num_blocked_by_max_distance"] = statistics.get("num_blocked_by_max_distance", 0) + 1
                    document.attribute_mappings[attribute.name] = []

            tak: float = time.time()
            logger.info(f"Updated remaining documents in {tak - tik} seconds.")

        tack: float = time.time()
        logger.info(f"Executed matching phase '{self.matching_phase_str}' on document base with "
                    f"{len(document_base.documents)} documents and {len(document_base.attributes)} attributes "
                    f"in {tack - tick} seconds.")

    def to_config(self) -> Dict[str, Any]:
        return {
            "matching_phase_str": self.matching_phase_str,
            "distance": self._distance.to_config(),
            "max_num_feedback": self._max_num_feedback,
            "len_ranked_list": self._len_ranked_list,
            "max_distance": self._max_distance
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RankingBasedMatchingPhase":
        distance: BaseDistance = BaseDistance.from_config(config["distance"])
        return cls(distance, config["max_num_feedback"], config["len_ranked_list"], config["max_distance"])
