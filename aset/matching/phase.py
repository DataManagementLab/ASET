import abc
import logging
import random
import time
from typing import Any, Callable, Dict, List, Type

import numpy as np
from scipy.special import softmax

from aset.config import ConfigurableElement
from aset.data.annotations import CurrentMatchIndexAnnotation, SentenceStartCharsAnnotation
from aset.data.data import ASETDocument, ASETDocumentBase, ASETNugget
from aset.data.signals import CachedContextSentenceSignal, CachedDistanceSignal, TreePredecessorSignal, \
    UserProvidedExamplesSignal
from aset.matching.distance import BaseDistance
from aset.preprocessing.embedding import BaseEmbedder
from aset.statistics import Statistics
from aset.status import StatusFunction

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

    def __hash__(self) -> int:
        return hash(self.matching_phase_str)

    @abc.abstractmethod
    def __call__(
            self,
            document_base: ASETDocumentBase,
            feedback_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        """
        Find matching ASETNuggets for the ASETAttributes.

        :param document_base: document base to work on
        :param feedback_fn: callback function for feedback
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
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

    def __call__(
            self,
            document_base: ASETDocumentBase,
            feedback_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        logger.info(
            f"Execute matching phase '{self.matching_phase_str}' on document base with "
            f"{len(document_base.documents)} documents and {len(document_base.attributes)} attributes."
        )
        tick: float = time.time()
        status_fn(f"Running {self.matching_phase_str}...", -1)

        statistics["matching_phase_str"] = self.matching_phase_str
        statistics["num_documents"] = len(document_base.documents)
        statistics["num_nuggets"] = len(document_base.nuggets)

        # incorporate examples
        for attribute in document_base.attributes:
            feedback_result: Dict[str, Any] = feedback_fn({"message": "give-examples", "attribute": attribute})
            examples: List[str] = feedback_result["examples"]
            statistics["examples"][attribute.name] = examples
            attribute[UserProvidedExamplesSignal] = UserProvidedExamplesSignal(examples)

        self._examples_embedder(document_base, status_fn, statistics["examples_embedder"])

        # execute distance
        self._distance(document_base, status_fn, statistics["distance"])

        # match attribute-by-attribute
        for attribute in document_base.attributes:
            logger.info(f"Match attribute '{attribute.name}'.")
            tik: float = time.time()
            self._distance.next_attribute()

            num_feedback: int = 0

            remaining_nuggets: List[ASETNugget] = []
            matching_nuggets: List[ASETNugget] = []
            queue: List[ASETNugget] = []

            # initialize the distances with the distances to the attribute
            distances_to_attribute: bool = True
            if document_base.nuggets != []:

                distances: np.ndarray = self._distance.compute_distances(
                    [attribute], document_base.nuggets, statistics["distance"]
                )

                for nugget, distance in zip(document_base.nuggets, distances[0]):
                    nugget[CachedDistanceSignal] = CachedDistanceSignal(distance)
                    remaining_nuggets.append(nugget)

            while num_feedback < self._max_num_feedback:

                # find a root
                root_iteration: int = 1
                while queue == [] and num_feedback < self._max_num_feedback:
                    temperature: float = 0.001 * (root_iteration ** 2)
                    weights: List[float] = [
                        (1 - nugget[CachedDistanceSignal]) / temperature for nugget in remaining_nuggets
                    ]
                    softmax_weights: np.ndarray = softmax(weights)

                    nugget: ASETNugget = random.choices(remaining_nuggets, weights=softmax_weights)[0]

                    num_feedback += 1
                    feedback_result: Dict[str, Any] = feedback_fn(
                        {
                            "message": "give-feedback",
                            "nugget": nugget,
                            "attribute": attribute
                        }
                    )
                    if feedback_result["feedback"]:
                        statistics["attributes"][attribute.name]["positive_feedback"] += 1
                        remaining_nuggets: List[ASETNugget] = [
                            n for n in remaining_nuggets if n.document is not nugget.document
                        ]
                        matching_nuggets.append(nugget)
                        queue.append(nugget)
                    else:
                        statistics["attributes"][attribute.name]["negative_feedback"] += 1
                    root_iteration += 1

                # if the distances are still the distances to the attribute, set them to one
                if distances_to_attribute:
                    for nugget in remaining_nuggets:
                        nugget[CachedDistanceSignal] = 1
                    distances_to_attribute = False

                # explore the tree
                while queue != [] and num_feedback < self._max_num_feedback:
                    nugget = queue.pop(0)  # pop the first element ==> FIFO queue

                    # compute the distance of the current node to all other already expanded nodes
                    distance: float = 1
                    for nug in matching_nuggets:
                        if nug is not nugget:
                            dist: float = self._distance.compute_distance(nugget, nug, statistics["distance"])
                            if dist < distance:
                                distance: float = dist

                    if len(matching_nuggets) == 1:
                        distance: float = 0

                    # compute the distances to the current, update distances if necessary and find possible samples
                    samples: List[ASETNugget] = []
                    if remaining_nuggets != []:
                        new_distances: np.ndarray = self._distance.compute_distances(
                            [nugget], remaining_nuggets, statistics["distance"]
                        )
                        for nug, new_dist in zip(remaining_nuggets, new_distances[0]):
                            # explore only if closer to this one than to any other one
                            if new_dist < nug[CachedDistanceSignal]:
                                nug[CachedDistanceSignal] = new_dist
                                if new_dist / self._exploration_factor > distance:  # explore farther away
                                    samples.append(nug)

                    samples: List[ASETNugget] = sorted(samples, key=lambda x: x[CachedDistanceSignal])
                    samples: List[ASETNugget] = samples[: self._max_children]

                    # query the user about the samples
                    new_matching: List[ASETNugget] = []
                    for nug in samples:
                        if num_feedback < self._max_num_feedback:
                            num_feedback += 1
                            feedback_result: Dict[str, Any] = feedback_fn({
                                "message": "give-feedback",
                                "nugget": nug,
                                "attribute": attribute
                            })
                            if feedback_result["feedback"]:
                                statistics["attributes"][attribute.name]["positive_feedback"] += 1
                                matching_nuggets.append(nug)
                                new_matching.append(nug)
                            else:
                                statistics["attributes"][attribute.name]["negative_feedback"] += 1

                    # update queue and remaining
                    queue += new_matching
                    for nug in new_matching:
                        remaining_nuggets: List[ASETNugget] = [
                            n for n in remaining_nuggets if n.document is not nug.document
                        ]

            # update the distances with the remaining stack
            for nugget in queue:
                if remaining_nuggets != []:
                    new_distances: np.ndarray = self._distance.compute_distances(
                        [nugget], remaining_nuggets, statistics["distance"]
                    )
                    for nug, new_dist in zip(remaining_nuggets, new_distances[0]):
                        if new_dist < nug[CachedDistanceSignal]:
                            nug[CachedDistanceSignal] = new_dist

            # match based on the calculated distances
            for nugget in matching_nuggets:
                statistics["attributes"][attribute.name]["confirmed_matches"] += 1
                nugget.document.attribute_mappings[attribute.name] = [nugget]
                nugget[CachedDistanceSignal] = 0

            for nugget in remaining_nuggets:
                distance: float = nugget[CachedDistanceSignal]
                if attribute.name not in nugget.document.attribute_mappings.keys():
                    nugget.document.attribute_mappings[attribute.name] = []
                existing_nuggets: List[ASETNugget] = nugget.document.attribute_mappings[attribute.name]

                if existing_nuggets == [] or (
                        distance < existing_nuggets[0][CachedDistanceSignal]
                        and distance < self._max_distance
                ):
                    if existing_nuggets == []:
                        statistics["attributes"][attribute.name]["guessed_matches"] += 1
                    nugget.document.attribute_mappings[attribute.name] = [nugget]

            tak: float = time.time()
            logger.info(f"Matched attribute '{attribute.name}' in {tak - tik} seconds.")

        status_fn(f"Running {self.matching_phase_str}...", 1)
        tack: float = time.time()
        logger.info(
            f"Executed matching phase '{self.matching_phase_str}' on document base with "
            f"{len(document_base.documents)} documents and {len(document_base.attributes)} attributes "
            f"in {tack - tick} seconds."
        )

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

    works with signals: CachedContextSentenceSignal, CachedDistanceSignal, TreePredecessorSignal
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
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        logger.info(
            f"Execute matching phase '{self.matching_phase_str}' on document base with "
            f"{len(document_base.documents)} documents and {len(document_base.attributes)} attributes."
        )
        tick: float = time.time()
        status_fn(f"Running {self.matching_phase_str}...", -1)

        statistics["matching_phase_str"] = self.matching_phase_str
        statistics["num_documents"] = len(document_base.documents)
        statistics["num_nuggets"] = len(document_base.nuggets)

        # execute distance
        self._distance(document_base, status_fn, statistics["distance"])

        # cache the context sentences
        logger.info("Cache the context sentences.")
        tik: float = time.time()

        for nugget in document_base.nuggets:
            sent_start_chars: List[int] = nugget.document[SentenceStartCharsAnnotation]
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

            nugget[CachedContextSentenceSignal] = CachedContextSentenceSignal({
                "text": context_sentence,
                "start_char": start_in_context,
                "end_char": end_in_context
            })

        tak: float = time.time()
        logger.info(f"Cached context sentences in {tak - tik} seconds.")

        for attribute in document_base.attributes:
            logger.info(f"Matching attribute '{attribute.name}'.")
            self._distance.next_attribute()
            remaining_documents: List[ASETDocument] = []

            # compute initial distances as distances to label
            logger.info("Compute initial distances and initialize documents.")
            tik: float = time.time()

            distances: np.ndarray = self._distance.compute_distances(
                [attribute], document_base.nuggets, statistics["distance"]
            )[0]
            for nugget, distance in zip(document_base.nuggets, distances):
                nugget[CachedDistanceSignal] = CachedDistanceSignal(distance)
                nugget[TreePredecessorSignal] = TreePredecessorSignal(None)

            for document in document_base.documents:
                index: int = -1
                dist: float = 2
                for ix, nugget in enumerate(document.nuggets):
                    if nugget[CachedDistanceSignal] < dist:
                        dist = nugget[CachedDistanceSignal]
                        index = ix
                if index != -1:
                    document[CurrentMatchIndexAnnotation] = CurrentMatchIndexAnnotation(index)
                    remaining_documents.append(document)
                else:  # document has no nuggets
                    document.attribute_mappings[attribute.name] = []
                    statistics[attribute.name]["num_document_with_no_nuggets"] += 1

            tak: float = time.time()
            logger.info(f"Computed initial distances and initialized documents in {tak - tik} seconds.")

            # iterative user interactions
            logger.info("Execute interactive matching.")
            tik: float = time.time()
            num_feedback: int = 0
            continue_matching: bool = True
            initial_distances: bool = True
            while continue_matching and num_feedback < self._max_num_feedback and remaining_documents != []:
                # sort remaining documents by distance
                remaining_documents = list(sorted(
                    remaining_documents,
                    key=lambda x: x.nuggets[x[CurrentMatchIndexAnnotation]][CachedDistanceSignal],
                    reverse=True
                ))

                # present documents to the user for feedback
                feedback_nuggets: List[ASETNugget] = []
                for doc in remaining_documents[:self._len_ranked_list]:
                    feedback_nuggets.append(doc.nuggets[doc[CurrentMatchIndexAnnotation]])
                num_feedback += 1
                feedback_result: Dict[str, Any] = feedback_fn({"nuggets": feedback_nuggets, "attribute": attribute})

                if feedback_result["message"] == "stop-interactive-matching":
                    statistics[attribute.name]["stopped_matching_by_hand"] = True
                    continue_matching = False
                elif feedback_result["message"] == "no-match-in-document":
                    statistics[attribute.name]["num_no_match_in_document"] += 1
                    feedback_result["nugget"].document.attribute_mappings[attribute.name] = []
                    remaining_documents.remove(feedback_result["nugget"].document)

                    # give feedback
                    for nugget in feedback_result["nugget"].document.nuggets:
                        if nugget[TreePredecessorSignal] is not None:
                            self._distance.feedback_no_match(
                                nugget,
                                nugget[TreePredecessorSignal],
                                statistics["distance"],
                            )

                elif feedback_result["message"] == "is-match":
                    statistics[attribute.name]["num_confirmed_match"] += 1
                    feedback_result["nugget"].document.attribute_mappings[attribute.name] = [feedback_result["nugget"]]
                    remaining_documents.remove(feedback_result["nugget"].document)

                    # give feedback
                    if feedback_result["nugget"][TreePredecessorSignal] is not None:
                        self._distance.feedback_match(
                            feedback_result["nugget"],
                            feedback_result["nugget"][TreePredecessorSignal],
                            statistics["distance"],
                        )

                    # update the distances for the other documents
                    for document in remaining_documents:
                        new_distances: np.ndarray = self._distance.compute_distances(
                            [feedback_result["nugget"]],
                            document.nuggets,
                            statistics["distance"]
                        )[0]
                        for nugget, new_distance in zip(document.nuggets, new_distances):
                            if initial_distances or new_distance < nugget[CachedDistanceSignal]:
                                nugget[CachedDistanceSignal] = new_distance
                                nugget[TreePredecessorSignal] = feedback_result["nugget"]
                        initial_distances = False
                        for ix, nugget in enumerate(document.nuggets):
                            current_guess: ASETNugget = document.nuggets[document[CurrentMatchIndexAnnotation]]
                            if nugget[CachedDistanceSignal] < current_guess[CachedDistanceSignal]:
                                document[CurrentMatchIndexAnnotation] = ix
            tak: float = time.time()
            logger.info(f"Executed interactive matching in {tak - tik} seconds.")

            # update remaining documents
            logger.info("Update remaining documents.")
            tik: float = time.time()

            for document in remaining_documents:
                current_guess: ASETNugget = document.nuggets[document[CurrentMatchIndexAnnotation]]
                if current_guess[CachedDistanceSignal] < self._max_distance:
                    statistics[attribute.name]["num_guessed_match"] += 1
                    document.attribute_mappings[attribute.name] = [current_guess]
                else:
                    statistics[attribute.name]["num_blocked_by_max_distance"] += 1
                    document.attribute_mappings[attribute.name] = []

            tak: float = time.time()
            logger.info(f"Updated remaining documents in {tak - tik} seconds.")

        status_fn(f"Running {self.matching_phase_str}...", 1)
        tack: float = time.time()
        logger.info(
            f"Executed matching phase '{self.matching_phase_str}' on document base with "
            f"{len(document_base.documents)} documents and {len(document_base.attributes)} attributes "
            f"in {tack - tick} seconds."
        )

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
