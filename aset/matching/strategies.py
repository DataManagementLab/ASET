"""Matching strategies to match between extractions and attributes."""
import logging
from abc import ABC, abstractmethod
from operator import itemgetter
from random import choices
from typing import Callable

from scipy.special import softmax

from aset.core.resources import close_all_resources
from aset.extraction.common import Document, Extraction
from aset.matching.common import Attribute, Row

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Strategy to match extractions to attributes. Super class for all matching strategies."""

    strategy_str = "BaseStrategy"

    def __eq__(self, other):
        return self.strategy_str == other.strategy_str

    @abstractmethod
    def __call__(self, documents: [Document], attributes: [Attribute]):
        """Match extractions from the given documents to the given attributes and return a list of filled rows."""
        raise NotImplementedError


class StaticMatching(BaseStrategy):
    """StaticMatching strategy."""

    strategy_str = "StaticMatching"

    def __init__(self, max_distance: float):
        """
        Initialize matching strategy.

        :param max_distance: maximum distance to accept extractions from in the final matching process
        """
        super(StaticMatching, self).__init__()
        self.max_distance = max_distance

    def __call__(self, documents: [Document], attributes: [Attribute]):
        """Match extractions from the given documents to the given attributes and return a list of filled rows."""

        rows = [Row(attributes) for _ in documents]

        for row, document in zip(rows, documents):
            for attribute in attributes:

                closest_dist = 1
                for extraction in document.extractions:
                    dist = attribute.embedding.distance(extraction.embedding)
                    if dist < closest_dist and dist < self.max_distance:
                        row.extractions[attribute.label] = extraction
                        closest_dist = dist

        return rows


class TreeSearchExploration(BaseStrategy):
    """TreeSearchExploration matching strategy."""

    strategy_str = "TreeSearchExploration"

    def __init__(self,
                 max_roots: int,
                 max_initial_tries: int,
                 max_children: int,
                 explore_far_factor: float,
                 max_distance: float,
                 max_interactions: int
                 ):
        """
        Initialize matching strategy.

        :param max_roots: number of extractions to find using just the attribute label
        :param max_initial_tries: maximum number of tries to find the set of initial matching extractions
        :param max_children: number of extractions to explore in every step
        :param explore_far_factor: extraction must be closer than previous distance / explore_far_factor to be sampled
        :param max_distance: maximum distance to accept extractions from
        :param max_interactions: maximum number of user interactions
        """
        super(TreeSearchExploration, self).__init__()

        self.max_initial_tries: int = max_initial_tries
        self.max_roots: int = max_roots
        self.max_children: int = max_children
        self.max_distance: float = max_distance
        self.exploration_factor: float = explore_far_factor
        self.max_interactions: int = max_interactions

    def __call__(self, documents: [Document], attributes: [Attribute]):
        """Match extractions from the given documents to the given attributes and return a list of filled rows."""

        rows = [Row(attributes) for _ in documents]

        for attribute in attributes:
            print("\n\n")
            logger.debug("Match attribute '{}'.".format(attribute.label))

            num_interactions = 0

            # 1. find a set of initial matching extractions
            remaining = []  # (document index, extraction, distance)
            matching_extractions = []  # (document index, extraction)

            # initialize the distances with the distances to the attribute
            weights = []
            for i, document in enumerate(documents):
                for extraction in document.extractions:
                    distance = attribute.embedding.distance(extraction.embedding)
                    remaining.append((i, extraction, distance))
                    weights.append(1 - distance)

            # sample extractions with rising temperature and present them to the user
            while len(matching_extractions) < self.max_roots \
                    and (matching_extractions == [] or num_interactions < self.max_initial_tries) \
                    and num_interactions < self.max_interactions:
                temperature = 0.001 * (num_interactions + 1) * (num_interactions + 1)
                softmax_weights = softmax([weight / temperature for weight in weights])

                document, extraction, distance = choices(remaining, weights=softmax_weights)[0]
                num_interactions += 1
                is_add_attribute = yield document, attribute, extraction, num_interactions
                if is_add_attribute:
                    matching_extractions.append((document, extraction))
                    new_remaining = []
                    new_weights = []
                    for weight, (doc, ext, dist) in zip(weights, remaining):
                        if doc != document:  # throw out extractions from the document
                            new_remaining.append((doc, ext, dist))
                            new_weights.append(weight)
                    remaining = new_remaining
                    weights = new_weights

            # 2. explore the embedding space
            stack = [(document, extraction, 0) for document, extraction in matching_extractions]

            # initialize all distances as the shortest distance to one in the initial set of matching extractions
            new_remaining = []
            for document, extraction, distance in remaining:
                new_distance = 1
                for doc, ext in matching_extractions:
                    dist = extraction.embedding.distance(ext.embedding)
                    if dist < new_distance:
                        new_distance = dist
                new_remaining.append((document, extraction, 1))
            remaining = new_remaining

            # while the stack is not empty and the maximum number of steps is not reached
            while len(stack) > 0 and num_interactions < self.max_interactions:
                document, extraction, distance = stack.pop()
                # the popped distance is the distance of the current extraction to the tree at the time it was added

                # compute the distance of the current node to the rest of the tree
                distance = 1
                for doc, ext in matching_extractions:
                    if ext is not extraction:
                        dist = extraction.embedding.distance(ext.embedding)
                        if dist < distance:
                            distance = dist
                if len(matching_extractions) == 1:
                    distance = 0

                # compute the distances to the current, update distances if necessary and find possible samples
                samples = []
                new_remaining = []
                for doc, ext, dist in remaining:
                    new_dist = extraction.embedding.distance(ext.embedding)
                    if new_dist < dist:  # explore only if closer to this one than any other one
                        new_remaining.append((doc, ext, new_dist))
                        if dist / self.exploration_factor > distance:  # explore farther away
                            samples.append((doc, ext, new_dist))
                    else:
                        new_remaining.append((doc, ext, dist))
                remaining = new_remaining

                samples = sorted(samples, key=lambda x: x[2])
                samples = samples[:self.max_children]

                # query the user about the samples
                new_matching = []
                for doc, ext, dist in samples:
                    if num_interactions < self.max_interactions:
                        num_interactions += 1
                        is_add_attribute = yield doc, attribute, ext, num_interactions
                        if is_add_attribute:
                            matching_extractions.append((doc, ext))
                            new_matching.append((doc, ext, dist))

                # update the stack and remaining
                stack += [(doc, ext, dist) for doc, ext, dist in new_matching]
                docs_to_remove = [doc for doc, ext, dist in new_matching]
                if len(docs_to_remove) > 0:
                    new_remaining = []
                    for doc, ext, dist in remaining:
                        if doc not in docs_to_remove:
                            new_remaining.append((doc, ext, dist))
                    remaining = new_remaining

            # update the distances with the remaining stack
            for document, extraction, distance in stack:
                new_remaining = []
                for doc, ext, dist in remaining:
                    new_dist = extraction.embedding.distance(ext.embedding)
                    if new_dist < dist:
                        new_remaining.append((doc, ext, new_dist))
                    else:
                        new_remaining.append((doc, ext, dist))
                remaining = new_remaining

            # match based on the calculated distances
            closest_distances = [1] * len(rows)
            for document, extraction in matching_extractions:
                rows[document].extractions[attribute.label] = extraction
                closest_distances[document] = 0

            for document, extraction, distance in remaining:
                if distance < self.max_distance and distance < closest_distances[document]:
                    rows[document].extractions[attribute.label] = extraction
                    closest_distances[document] = distance

        return rows


class DFSExploration(BaseStrategy):
    """DFSExploration matching strategy."""

    strategy_str = "DFSExploration"

    def __init__(self,
                 max_children: int,
                 explore_far_factor: float,
                 max_distance: float,
                 max_interactions: int
                 ):
        """
        Initialize matching strategy.

        :param max_children: number of extractions to explore in every step
        :param explore_far_factor: extraction must be closer than previous distance / explore_far_factor to be sampled
        :param max_distance: maximum distance to accept extractions from
        :param max_interactions: maximum number of user interactions
        """
        super(DFSExploration, self).__init__()

        self.max_children: int = max_children
        self.max_distance: float = max_distance
        self.exploration_factor: float = explore_far_factor
        self.max_interactions: int = max_interactions

    def __call__(self, documents: [Document], attributes: [Attribute]):
        """Match extractions from the given documents to the given attributes and return a list of filled rows."""

        rows = [Row(attributes) for _ in documents]

        for attribute in attributes:
            print("\n\n")
            logger.debug("Match attribute '{}'.".format(attribute.label))

            num_interactions = 0

            remaining = []  # (document index, extraction, distance from already expanded nodes)
            matching_extractions = []  # (document index, extraction)
            queue = []  # (document index, extraction)

            # initialize the distances with the distances to the attribute
            distances_to_attribute = True
            for document_index, document in enumerate(documents):
                for extraction in document.extractions:
                    distance = attribute.embedding.distance(extraction.embedding)
                    remaining.append((document_index, extraction, distance))

            while num_interactions < self.max_interactions:

                # find a root
                root_iteration = 1
                while not queue and num_interactions < self.max_interactions:
                    temperature = 0.01 * (root_iteration ** 2)
                    weights = [(1 - distance) / temperature for _, _, distance in remaining]
                    softmax_weights = softmax(weights)

                    document_index, extraction, distance = choices(remaining, weights=softmax_weights)[0]

                    num_interactions += 1
                    is_add_attribute = yield document_index, attribute, extraction, num_interactions
                    if is_add_attribute:
                        remaining = list(filter(lambda x: x[0] != document_index, remaining))
                        matching_extractions.append((document_index, extraction))
                        queue.append((document_index, extraction))
                    root_iteration += 1

                # if the distances are still the distances to the attribute, set them to one
                if distances_to_attribute:
                    remaining = [(doc, ext, 1) for doc, ext, dist in remaining]
                    distances_to_attribute = False

                # explore the tree
                while queue and num_interactions < self.max_interactions:
                    document_index, extraction = queue.pop(0)  # pop the first element ==> FIFO queue

                    # compute the distance of the current node to all other already expanded nodes
                    distance = 1
                    for doc, ext in matching_extractions:
                        if ext is not extraction:
                            dist = extraction.embedding.distance(ext.embedding)
                            if dist < distance:
                                distance = dist

                    if len(matching_extractions) == 1:
                        distance = 0

                    # compute the distances to the current, update distances if necessary and find possible samples
                    samples = []
                    new_remaining = []
                    for doc, ext, dist in remaining:
                        new_dist = extraction.embedding.distance(ext.embedding)
                        if new_dist < dist:  # explore only if closer to this one than to any other one
                            new_remaining.append((doc, ext, new_dist))
                            if dist / self.exploration_factor > distance:  # explore farther away
                                samples.append((doc, ext, new_dist))
                        else:
                            new_remaining.append((doc, ext, dist))
                    remaining = new_remaining

                    samples = sorted(samples, key=itemgetter(2))
                    samples = samples[:self.max_children]

                    # query the user about the samples
                    new_matching = []
                    for doc, ext, dist in samples:
                        if num_interactions < self.max_interactions:
                            num_interactions += 1
                            is_add_attribute = yield doc, attribute, ext, num_interactions
                            if is_add_attribute:
                                matching_extractions.append((doc, ext))
                                new_matching.append((doc, ext))

                    # update queue and remaining
                    queue += new_matching
                    docs_to_remove = set(doc for doc, _ in new_matching)
                    if docs_to_remove:
                        remaining = list(filter(lambda x: x[0] not in docs_to_remove, remaining))

            # update the distances with the remaining stack
            for document_index, extraction in queue:
                new_remaining = []
                for doc, ext, dist in remaining:
                    new_dist = extraction.embedding.distance(ext.embedding)
                    if new_dist < dist:
                        new_remaining.append((doc, ext, new_dist))
                    else:
                        new_remaining.append((doc, ext, dist))
                remaining = new_remaining

            # match based on the calculated distances
            closest_distances = [1] * len(rows)
            for document_index, extraction in matching_extractions:
                rows[document_index].extractions[attribute.label] = extraction
                closest_distances[document_index] = 0

            for document_index, extraction, distance in remaining:
                if distance < self.max_distance and distance < closest_distances[document_index]:
                    rows[document_index].extractions[attribute.label] = extraction
                    closest_distances[document_index] = distance

        return rows


def query_user(document_index: int, attribute: Attribute, extraction: Extraction, num_interactions: int):
    """
    Ask the user whether the given extraction belongs to the given attribute.

    :param document_index: index of the document
    :param attribute: given attribute
    :param extraction: given extraction
    :param num_interactions: number of user interactions including this one
    :return: True if the extraction belongs to the attribute, else False
    """
    print("\n{:4.4}  '{}'?   '{}'   from   '{}' from {}".format(
        str(num_interactions) + ".",
        attribute.label,
        extraction.mention,
        extraction.context.replace("\n", " "),
        str(document_index)
    ))

    while True:
        s = input("y/n: ")
        if s == "y":
            return True
        elif s == "n":
            return False
        elif s == "close":
            close_all_resources()
            exit()
