import abc
import logging
import random
import time
from typing import Any, Dict, List

import numpy as np
from scipy.special import softmax

from aset.configuration import BasePipelineElement, register_configurable_element
from aset.data.data import ASETDocument, ASETDocumentBase, ASETNugget
from aset.data.signals import CachedContextSentenceSignal, CachedDistanceSignal, TreePredecessorSignal, \
    UserProvidedExamplesSignal, SentenceStartCharsSignal, CurrentMatchIndexSignal
from aset.interaction import BaseInteractionCallback
from aset.matching.distance import BaseDistance
from aset.preprocessing.embedding import BaseEmbedder
from aset.statistics import Statistics
from aset.status import BaseStatusCallback

logger: logging.Logger = logging.getLogger(__name__)


class BaseMatcher(BasePipelineElement, abc.ABC):
    """
    Base class for all matchers.

    A matcher attempts to find matching ASETNuggets for the ASETAttributes.
    """
    identifier: str = "BaseMatcher"


########################################################################################################################
# actual matchers
########################################################################################################################


@register_configurable_element
class RankingBasedMatcher(BaseMatcher):
    """Matcher that displays a ranked list of nuggets to the user for feedback."""

    identifier: str = "RankingBasedMatcher"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [CachedContextSentenceSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [
            CachedDistanceSignal.identifier,
            TreePredecessorSignal.identifier
        ],
        "attributes": [],
        "documents": []
    }

    def __init__(
            self,
            distance: BaseDistance,
            max_num_feedback: int,
            len_ranked_list: int,
            max_distance: float,
            num_random_docs: int,
            sampling_mode: str
    ) -> None:
        """
        Initialize the RankingBasedMatcher.

        :param distance: distance function
        :param max_num_feedback: maximum number of user interactions per attribute
        :param len_ranked_list: length of the ranked list of nuggets presented to the user for feedback
        :param max_distance: maximum distance at which nuggets will be accepted
        :param num_random_docs: number of random documents that are part of the ranked list of nuggets
        :param sampling_mode: determines how to sample the nuggets presented to the user for feedback
        """
        super(RankingBasedMatcher, self).__init__()
        self._distance: BaseDistance = distance
        self._max_num_feedback: int = max_num_feedback
        self._len_ranked_list: int = len_ranked_list
        self._max_distance: float = max_distance
        self._num_random_docs: int = num_random_docs
        self._sampling_mode: str = sampling_mode

        # add signals required by the distance function to the signals required by the matcher
        self._add_required_signal_identifiers(self._distance.required_signal_identifiers)

        logger.debug(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document_base: ASETDocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["num_documents"] = len(document_base.documents)
        statistics["num_nuggets"] = len(document_base.nuggets)

        for attribute in document_base.attributes:
            logger.info(f"Matching attribute '{attribute.name}'.")
            self._distance.next_attribute()

            already_matched = False
            for document in document_base.documents:
                if attribute.name in document.attribute_mappings.keys():
                    already_matched = True
            if already_matched:
                logger.info(f"Attribute '{attribute.name}' has already been matched before.")
                continue

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
                    document[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(index)
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
                    key=lambda x: x.nuggets[x[CurrentMatchIndexSignal]][CachedDistanceSignal],
                    reverse=True
                ))

                if self._sampling_mode == "MOST_UNCERTAIN":
                    selected_documents: List[ASETDocument] = remaining_documents[:self._len_ranked_list]
                elif self._sampling_mode == "MOST_UNCERTAIN_WITH_RANDOMS":
                    # sample random documents and move them to the front of the ranked list
                    random_documents: List[ASETDocument] = []
                    for i in range(self._num_random_docs):
                        if remaining_documents != []:
                            random_document: ASETDocument = random.choice(remaining_documents)
                            random_documents.append(random_document)
                            remaining_documents.remove(random_document)

                    remaining_documents: List[ASETDocument] = random_documents + remaining_documents
                    selected_documents = remaining_documents[:self._len_ranked_list]
                elif self._sampling_mode == "AT_MAX_DISTANCE_THRESHOLD":
                    ix_lower: int = 0
                    while ix_lower < len(remaining_documents) and \
                            remaining_documents[ix_lower].nuggets[
                                remaining_documents[ix_lower][CurrentMatchIndexSignal]][
                                CachedDistanceSignal] > self._max_distance:
                        ix_lower += 1

                    higher_left: int = max(0, ix_lower - self._len_ranked_list // 2)
                    higher_right: int = ix_lower
                    higher_num: int = higher_right - higher_left
                    lower_left: int = ix_lower
                    lower_right: int = min(len(remaining_documents), ix_lower + self._len_ranked_list // 2)
                    lower_num: int = lower_right - lower_left

                    if lower_num < self._len_ranked_list // 2:
                        higher_left = max(0, higher_left - (self._len_ranked_list // 2 - lower_num))
                    elif higher_num < self._len_ranked_list // 2:
                        lower_right = min(len(remaining_documents),
                                          lower_right + (self._len_ranked_list // 2 - higher_num))

                    selected_documents: List[ASETDocument] = remaining_documents[higher_left:lower_right]
                else:
                    logger.error(f"Unknown sampling mode '{self._sampling_mode}'!")
                    assert False, f"Unknown sampling mode '{self._sampling_mode}'!"

                # present documents to the user for feedback
                feedback_nuggets: List[ASETNugget] = []
                for doc in selected_documents:
                    feedback_nuggets.append(doc.nuggets[doc[CurrentMatchIndexSignal]])
                num_feedback += 1
                feedback_result: Dict[str, Any] = interaction_callback(
                    self.identifier,
                    {
                        "max-distance": self._max_distance,
                        "nuggets": feedback_nuggets,
                        "attribute": attribute
                    }
                )

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
                            current_guess: ASETNugget = document.nuggets[document[CurrentMatchIndexSignal]]
                            if nugget[CachedDistanceSignal] < current_guess[CachedDistanceSignal]:
                                document[CurrentMatchIndexSignal] = ix
            tak: float = time.time()
            logger.info(f"Executed interactive matching in {tak - tik} seconds.")

            # update remaining documents
            logger.info("Update remaining documents.")
            tik: float = time.time()

            for document in remaining_documents:
                current_guess: ASETNugget = document.nuggets[document[CurrentMatchIndexSignal]]
                if current_guess[CachedDistanceSignal] < self._max_distance:
                    statistics[attribute.name]["num_guessed_match"] += 1
                    document.attribute_mappings[attribute.name] = [current_guess]
                else:
                    statistics[attribute.name]["num_blocked_by_max_distance"] += 1
                    document.attribute_mappings[attribute.name] = []

            tak: float = time.time()
            logger.info(f"Updated remaining documents in {tak - tik} seconds.")

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "distance": self._distance.to_config(),
            "max_num_feedback": self._max_num_feedback,
            "len_ranked_list": self._len_ranked_list,
            "max_distance": self._max_distance,
            "num_random_docs": self._num_random_docs,
            "sampling_mode": self._sampling_mode
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RankingBasedMatcher":
        distance: BaseDistance = BaseDistance.from_config(config["distance"])
        return cls(distance, config["max_num_feedback"], config["len_ranked_list"], config["max_distance"],
                   config["num_random_docs"], config["sampling_mode"])
