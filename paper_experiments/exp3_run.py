"""
Experiment 3
============
"""

# requires that the match_by_hand has been executed on all documents of the dataset

import logging.config
import random
import traceback
from collections import defaultdict
from os.path import isfile

import numpy as np

import aset.matching.strategies as strategies
from aset.core.resources import close_all_resources
from aset.embedding.aggregation import ExtractionEmbeddingMethod, AttributeEmbeddingMethod
from aset.extraction.common import Document, Extraction
from aset.extraction.extractionstage import ExtractionStage
from aset.extraction.extractors import StanzaExtractor as Extractor
from aset.matching.common import Attribute
from aset.matching.matchingstage import MatchingStage
from datasets.aviation import aviation as dataset

logging.config.fileConfig("../logging.conf", disable_existing_loggers=False)
logger = logging.getLogger()


def do_evaluation_run(
        run: int,
        random_seed: int,
        documents: [],
        column_names: [str],
        column_name2attribute_name: {}
):
    """Evaluate the system with the given random seed and parameters."""
    print("\n\n\nExecuting run {}.".format(run + 1))

    # load the extraction stage
    with open("cache.json", "r", encoding="utf-8") as file:
        aset_extraction_stage = ExtractionStage.from_json_str(file.read())

    # set the random seed
    random.seed(random_seed)

    # randomly select "user-provided" example mentions
    # the mentions are sampled from the mentions in the annotated dataset (not from the hand-matched extractions)
    num_example_mentions = 3
    example_mentions = []
    for attribute in dataset.ATTRIBUTES:

        possible_mentions = []
        for document in documents:
            possible_mentions += [m["mention"] for m in document["mentions"][attribute]]
            possible_mentions += [m["mention"] for m in document["mentions_diff_value"][attribute]]

        example_mentions_here = []
        while len(example_mentions_here) < num_example_mentions and possible_mentions:
            example_mention = random.choice(possible_mentions)
            possible_mentions.remove(example_mention)
            example_mentions_here.append(example_mention)

        example_mentions.append(example_mentions_here)

    print("Randomly chosen example mentions:")
    print(example_mentions)

    # create the ASET columns
    aset_attributes = [Attribute(column_name) for column_name in column_names]

    # engage the matching stage
    aset_matching_stage = MatchingStage(
        documents=aset_extraction_stage.documents,
        attributes=aset_attributes,
        # strategy=strategies.StaticMatching(
        #     max_distance=0.3
        # ),
        # strategy=strategies.TreeSearchExploration(
        #     max_roots=2,
        #     max_initial_tries=10,
        #     max_children=2,
        #     explore_far_factor=1.15,
        #     max_distance=0.3,
        #     max_interactions=25
        # ),
        strategy=strategies.DFSExploration(
            max_children=2,
            explore_far_factor=1.15,
            max_distance=0.3,
            max_interactions=25
        ),
        embedding_method=AttributeEmbeddingMethod()
    )
    aset_matching_stage.compute_attribute_embeddings()
    aset_matching_stage.incorporate_example_mentions(example_mentions)

    def automatic_query_user(document_index: int, attribute: Attribute, extraction: Extraction, num_user_queries: int):
        """Automation of the user interaction using the hand-matched dataset."""
        attribute_name = column_name2attribute_name[attribute.label]
        document = documents[document_index]

        # FINDME: select which mentions should be approved by the simulated user
        indices = document["evaluation"][Extractor.extractor_str]["mentions"][attribute_name]
        indices += document["evaluation"][Extractor.extractor_str]["mentions_diff_value"][attribute_name]

        extractions = [document["evaluation"][Extractor.extractor_str]["all_extractions"][i] for i in indices]
        mentions = [extraction["mention"] for extraction in extractions]
        match = extraction.mention in mentions

        print("[{}] {:3.3} '{}'?  '{}'  ==>  {}".format(
            str(run + 1),
            str(num_user_queries) + ".",
            attribute.label,
            extraction.mention,
            "y" if match else "n"
        ))

        return match

    generator = aset_matching_stage.match_extractions_to_attributes()
    document, attribute, extraction, num_interactions = next(generator)
    while True:
        try:
            is_match = automatic_query_user(document, attribute, extraction, num_interactions)
            document, attribute, extraction, num_interactions = generator.send(is_match)
        except StopIteration:
            break

    # evaluate the matching process
    recalls = {}
    precisions = {}
    f1_scores = {}

    recalls_diff_value = {}
    precisions_diff_value = {}
    f1_scores_diff_value = {}

    for column_name, attribute_name in zip(column_names, dataset.ATTRIBUTES):
        num_mentioned = 0

        num_should_be_empty_is_empty = 0
        num_should_be_empty_is_full = 0

        num_should_be_filled_is_empty = 0
        num_should_be_filled_is_correct = 0
        num_should_be_filled_is_incorrect = 0

        for document, row in zip(documents, aset_matching_stage.rows):

            if document["mentions"][attribute_name]:  # document states cell's value

                num_mentioned += 1

                found_extraction = row.extractions[column_name]

                # find the valid mentions
                indices = document["evaluation"][Extractor.extractor_str]["mentions"][attribute_name]
                extractions = [document["evaluation"][Extractor.extractor_str]["all_extractions"][i] for i in indices]
                valid_mentions = [extraction["mention"] for extraction in extractions]

                if found_extraction is None:
                    num_should_be_filled_is_empty += 1
                elif found_extraction.mention in valid_mentions:
                    num_should_be_filled_is_correct += 1
                else:
                    num_should_be_filled_is_incorrect += 1

            else:  # document does not state cell's value
                found_extraction = row.extractions[column_name]
                if found_extraction is None:
                    num_should_be_empty_is_empty += 1
                else:
                    num_should_be_empty_is_full += 1

        # compute the evaluation metrics
        recall = num_should_be_filled_is_correct / (
                    num_should_be_filled_is_correct + num_should_be_filled_is_incorrect + num_should_be_filled_is_empty)

        if (num_should_be_filled_is_correct + num_should_be_filled_is_incorrect) == 0:
            precision = 1
        else:
            precision = num_should_be_filled_is_correct / (
                        num_should_be_filled_is_correct + num_should_be_filled_is_incorrect)

        f1_score = 2 * precision * recall / (precision + recall)

        recalls[attribute_name] = recall
        precisions[attribute_name] = precision
        f1_scores[attribute_name] = f1_score
        recalls_diff_value[attribute_name] = -1
        precisions_diff_value[attribute_name] = -1
        f1_scores_diff_value[attribute_name] = -1

    return recalls, precisions, f1_scores, recalls_diff_value, precisions_diff_value, f1_scores_diff_value


if __name__ == "__main__":
    try:
        # load the dataset and create the ASET documents
        documents = dataset.load_dataset()

        aset_documents = []
        for document in documents:
            aset_document = Document(document["text"])
            extraction_dicts = document["evaluation"][Extractor.extractor_str]["all_extractions"]
            aset_document.extractions = [Extraction.from_json_dict(d) for d in extraction_dicts]
            aset_documents.append(aset_document)

        # select the "user-provided" attribute names and create mappings between them and the dataset's attribute names
        attribute_names = [  # order must match the order of attributes in the dataset
            "event date",
            "city",
            "state",
            "airport code",
            "airport",
            "aircraft damage",
            "registration number",
            "manufacturer",
            "model",
            "far description",
            "airline",
            "weather condition"
        ]
        # attribute_names = [  # order must match the order of attributes in the dataset
        #     "date",
        #     "new cases",
        #     "new deaths",
        #     "incidence",
        #     "patients intensive care",
        #     "vaccinated",
        #     "twice vaccinated"
        # ]

        column_name2attribute_name = {
            column_name: attribute_name for column_name, attribute_name in zip(attribute_names, dataset.ATTRIBUTES)
        }

        # engage the extraction stage
        if not isfile("cache.json"):
            aset_extraction_stage = ExtractionStage(
                documents=aset_documents,
                extractors=[Extractor()],
                processors=[],  # skip the processing since it is not evaluated
                embedding_method=ExtractionEmbeddingMethod()
            )
            # do not derive extractions again, but use the loaded ones
            # skip determining values since it is not evaluated
            aset_extraction_stage.compute_extraction_embeddings()

            with open("cache.json", "w", encoding="utf-8") as file:
                file.write(aset_extraction_stage.json_str)

        # max_interactions = list(range(1, 51))
        # num_examples = list(range(0, 11))
        # max_distances = list(x / 40 for x in range(0, 41))
        # max_roots = [1, 2, 3, 4]
        # max_children = [2, 3, 4, 5, 6]
        # explore_far_factors = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
        # max_distances = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

        # for param_value in max_distances:
        #     print("###############################################")
        #     print("param_value:", param_value)
        #     print("###############################################")

        # evaluate the matching stage
        recalls, recalls_median = defaultdict(list), {}
        precisions, precisions_median = defaultdict(list), {}
        f1_scores, f1_scores_median = defaultdict(list), {}

        recalls_diff_value, recalls_diff_value_median = defaultdict(list), {}
        precisions_diff_value, precisions_diff_value_median = defaultdict(list), {}
        f1_scores_diff_value, f1_scores_diff_value_median = defaultdict(list), {}

        # random seeds have been randomly chosen once from [0, 1000000]
        random_seeds = [200488, 422329, 449756, 739608, 983889, 836016, 264198, 908457, 205619, 461905]

        # do the matching stage evaluation runs
        for run, random_seed in enumerate(random_seeds):
            result = do_evaluation_run(
                run,
                random_seed,
                documents,
                attribute_names,
                column_name2attribute_name
            )

            for attribute in dataset.ATTRIBUTES:
                recalls[attribute].append(result[0][attribute])
                precisions[attribute].append(result[1][attribute])
                f1_scores[attribute].append(result[2][attribute])
                recalls_diff_value[attribute].append(result[3][attribute])
                precisions_diff_value[attribute].append(result[4][attribute])
                f1_scores_diff_value[attribute].append(result[5][attribute])

        # compute the results as the median
        for attribute in dataset.ATTRIBUTES:
            recalls_median[attribute] = np.median(recalls[attribute])
            precisions_median[attribute] = np.median(precisions[attribute])
            f1_scores_median[attribute] = np.median(f1_scores[attribute])
            recalls_diff_value_median[attribute] = np.median(recalls_diff_value[attribute])
            precisions_diff_value_median[attribute] = np.median(precisions_diff_value[attribute])
            f1_scores_diff_value_median[attribute] = np.median(f1_scores_diff_value[attribute])

        # display the results
        print("\nrecalls:")
        for attribute in dataset.ATTRIBUTES:
            print(attribute, recalls_median[attribute], sorted(recalls[attribute]))
        print("\nprecisions:")
        for attribute in dataset.ATTRIBUTES:
            print(attribute, precisions_median[attribute], sorted(precisions[attribute]))
        print("\nf1_scores:")
        for attribute in dataset.ATTRIBUTES:
            print(attribute, f1_scores_median[attribute], sorted(f1_scores[attribute]))
        print("\nrecalls_diff_value:")
        for attribute in dataset.ATTRIBUTES:
            print(attribute, recalls_diff_value_median[attribute], sorted(recalls_diff_value[attribute]))
        print("\nprecisions_diff_value:")
        for attribute in dataset.ATTRIBUTES:
            print(attribute, precisions_diff_value_median[attribute], sorted(precisions_diff_value[attribute]))
        print("\nf1_scores_diff_value:")
        for attribute in dataset.ATTRIBUTES:
            print(attribute, f1_scores_diff_value_median[attribute], sorted(f1_scores_diff_value[attribute]))

        # store the results
        with open("../results/" + dataset.NAME + "/Stanza/end2end/" + "aset-dfs" + ".csv", "w") as file:
            file.write(
                "attribute, recall, precision, f1_score, recall_diff_value, precision_diff_value, f1_score_diff_value\n")

            for attribute in dataset.ATTRIBUTES:
                file.write(attribute + ", ")
                file.write(str(recalls_median[attribute]) + ", ")
                file.write(str(precisions_median[attribute]) + ", ")
                file.write(str(f1_scores_median[attribute]) + ", ")
                file.write(str(recalls_diff_value_median[attribute]) + ", ")
                file.write(str(precisions_diff_value_median[attribute]) + ", ")
                file.write(str(f1_scores_diff_value_median[attribute]) + "\n")

        close_all_resources()
    except:
        traceback.print_exc()
        close_all_resources()
