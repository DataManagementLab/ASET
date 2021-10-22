import json
import logging.config
import os
import random

import numpy as np

from aset.data.data import ASETDocumentBase, ASETDocument, ASETAttribute
from aset.matching.distance import SignalsMeanDistance
from aset.matching.phase import BaseMatchingPhase, RankingBasedMatchingPhase
from aset.preprocessing.embedding import SBERTTextEmbedder, RelativePositionEmbedder, FastTextLabelEmbedder, \
    BERTContextSentenceEmbedder
from aset.preprocessing.extraction import StanzaNERExtractor
from aset.preprocessing.phase import PreprocessingPhase
from aset.resources import ResourceManager
from aset.statistics import Statistics
from aset.status import StatusFunction
from datasets.aviation import aviation as dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def consider_overlap_as_match(x_start, x_end, y_start, y_end):
    """Determines whether the given spans x/y are considered as a match."""
    # considered as overlap if at least half of the smaller span
    x_length = x_end - x_start
    y_length = y_end - y_start
    valid_overlap = x_length // 2 if x_length > y_length else y_length // 2
    if x_start <= y_start:
        actual_overlap = min(x_end - y_start, y_length)
    else:
        actual_overlap = min(y_end - x_start, x_length)
    return actual_overlap >= valid_overlap


if __name__ == "__main__":
    with ResourceManager() as resource_manager:
        statistics = Statistics(do_collect=True)
        status_fn = StatusFunction(callback_fn=None)

        ################################################################################################################
        # dataset
        ################################################################################################################
        documents = dataset.load_dataset()

        statistics["dataset"]["dataset_name"] = dataset.NAME
        statistics["dataset"]["attributes"] = dataset.ATTRIBUTES
        statistics["dataset"]["num_documents"] = len(documents)

        for attribute in dataset.ATTRIBUTES:
            statistics["dataset"]["mentioned"][attribute] = 0
            for document in documents:
                if document["mentions"][attribute]:
                    statistics["dataset"]["mentioned"][attribute] += 1

        ################################################################################################################
        # document base
        ################################################################################################################
        # select the "user-provided" attribute names and create mappings between them and the dataset's attribute names
        attribute_names = dataset.ATTRIBUTES
        statistics["user_provided_attribute_names"] = attribute_names
        column_name2attribute_name = {
            column_name: attribute_name for column_name, attribute_name in zip(attribute_names, dataset.ATTRIBUTES)
        }

        document_base = ASETDocumentBase(
            documents=[ASETDocument(doc["id"], doc["text"]) for doc in documents],
            attributes=[ASETAttribute(attr_name) for attr_name in attribute_names]
        )

        ################################################################################################################
        # preprocessing phase
        ################################################################################################################
        if not os.path.isfile("../cache/unmatched-document-base.bson"):
            preprocessing_phase = PreprocessingPhase(
                extractors=[
                    StanzaNERExtractor()
                ],
                normalizers=[],
                embedders=[
                    FastTextLabelEmbedder("FastTextEmbedding100000", True, [" ", "_"]),
                    # SBERTLabelEmbedder("SBERTBertLargeNliMeanTokensResource"),
                    SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
                    BERTContextSentenceEmbedder("BertLargeCasedResource"),
                    # SBERTContextSentenceEmbedder("SBERTBertLargeNliMeanTokensResource"),
                    RelativePositionEmbedder()
                ]
            )

            with open("../cache/preprocessing-phase-config.json", "w") as file:
                json.dump(preprocessing_phase.to_config(), file)

            statistics["preprocessing"]["config"] = preprocessing_phase.to_config()

            preprocessing_phase(document_base, status_fn, statistics["preprocessing"]["statistics"])

            with open("../cache/unmatched-document-base.bson", "wb") as file:
                file.write(document_base.to_bson())
        else:
            with open("../cache/preprocessing-phase-config.json", "r") as file:
                statistics["preprocessing"]["config"] = json.load(file)
            with open("../cache/unmatched-document-base.bson", "rb") as file:
                document_base = ASETDocumentBase.from_bson(file.read())

        for attribute in dataset.ATTRIBUTES:
            statistics["preprocessing"]["results"][attribute] = 0
            for document, aset_document in zip(documents, document_base.documents):
                match = False
                for mention in document["mentions"][attribute]:
                    for nugget in aset_document.nuggets:
                        if consider_overlap_as_match(mention["start_char"], mention["end_char"],
                                                     nugget.start_char, nugget.end_char):
                            match = True
                            break
                if match:
                    statistics["preprocessing"]["results"][attribute] += 1

        ################################################################################################################
        # matching phase
        ################################################################################################################
        # matching_phase = TreeSearchMatchingPhase(
        #     distance=SignalsMeanDistance(
        #         signal_strings=[
        #             "LabelEmbeddingSignal",
        #             "TextEmbeddingSignal",
        #             "ContextSentenceEmbeddingSignal",
        #             "RelativePositionSignal",
        #             "POSTagsSignal"
        #         ]
        #     ),
        #     examples_embedder=SBERTExamplesEmbedder("SBERTBertLargeNliMeanTokensResource"),
        #     max_num_feedback=25,
        #     max_children=2,
        #     max_distance=0.6,
        #     exploration_factor=1.2
        # )
        matching_phase = RankingBasedMatchingPhase(
            distance=SignalsMeanDistance(
                signal_strings=[
                    "LabelEmbeddingSignal",
                    "TextEmbeddingSignal",
                    "ContextSentenceEmbeddingSignal",
                    "RelativePositionSignal"  # ,
                    # "POSTagsSignal"
                ]
            ),
            max_num_feedback=25,
            len_ranked_list=10,
            max_distance=0.6
        )

        with open("../cache/matching-phase-config.json", "w") as file:
            json.dump(matching_phase.to_config(), file)
        statistics["matching"]["config"] = matching_phase.to_config()

        statistics["matching"]["num_example_mentions"] = 3

        # random seeds have been randomly chosen once from [0, 1000000]
        random_seeds = [200488, 422329, 449756, 739608, 983889, 836016, 264198, 908457, 205619, 461905]
        for run, random_seed in enumerate(random_seeds):
            print("\n\n\nExecuting run {}.".format(run + 1))

            # load the document base
            with open("../cache/unmatched-document-base.bson", "rb") as file:
                document_base = ASETDocumentBase.from_bson(file.read())

            # set the random seed
            random.seed(random_seed)

            # randomly select "user-provided" example mentions
            # the mentions are sampled from the mentions in the annotated dataset
            example_mentions = {}
            for attribute, attribute_name in zip(dataset.ATTRIBUTES, attribute_names):

                possible_mentions = []
                for document in documents:
                    for mention in document["mentions"][attribute]:
                        possible_mentions.append(document["text"][mention["start_char"]:mention["end_char"]])
                    for mention in document["mentions_same_attribute_class"][attribute]:
                        possible_mentions.append(document["text"][mention["start_char"]:mention["end_char"]])

                sampled_mentions = []
                while len(sampled_mentions) < statistics["matching"]["num_example_mentions"] and possible_mentions:
                    example_mention = random.choice(possible_mentions)
                    possible_mentions.remove(example_mention)
                    sampled_mentions.append(example_mention)

                example_mentions[attribute_name] = sampled_mentions
                statistics["matching"]["runs"][str(run)]["example_mentions"] = example_mentions

            # engage the matching phase
            with open("../cache/matching-phase-config.json", "r") as file:
                matching_phase = BaseMatchingPhase.from_config(json.load(file))


            # for TreeSearchMatchingPhase
            # def automatic_feedback_fn(feedback_request):
            #     if feedback_request["message"] == "give-feedback":
            #         nug = feedback_request["nugget"]
            #         attr = feedback_request["attribute"]
            #
            #         attr_name = column_name2attribute_name[attr.name]
            #         doc = None
            #         for d in documents:
            #             if d["id"] == nug.document.name:
            #                 doc = d
            #
            #         match = False
            #         for mention in doc["mentions"][attr_name]:
            #             if consider_overlap_as_match(mention["start_char"], mention["end_char"],
            #                                          nug.start_char, nug.end_char):
            #                 match = True
            #                 break
            #         if not match:
            #             for mention in doc["mentions_same_attribute_class"][attr_name]:
            #                 if consider_overlap_as_match(mention["start_char"], mention["end_char"],
            #                                              nug.start_char, nug.end_char):
            #                     match = True
            #                     break
            #
            #         print("[{:3.3}] '{}'? '{}' ==> {}".format(
            #             str(run + 1),
            #             attr.name,
            #             nug.text,
            #             "yes" if match else "no"
            #         ))
            #
            #         return {"feedback": match}
            #
            #     elif feedback_request["message"] == "give-examples":
            #         attr = feedback_request["attribute"]
            #         return {"examples": example_mentions[attr.name]}
            #     else:
            #         assert False, f"Unknown message '{feedback_request['message']}'!"

            # for RankingBasedMatchingPhase
            def automatic_feedback_fn(feedback_request):
                nuggets = feedback_request["nuggets"]
                attr = feedback_request["attribute"]

                attr_name = column_name2attribute_name[attr.name]

                # user always gives feedback on first incorrect nugget guess if there is one
                for nug in nuggets:
                    doc = None
                    for d in documents:
                        if d["id"] == nug.document.name:
                            doc = d

                    for men in doc["mentions"][attr_name]:
                        if consider_overlap_as_match(nug.start_char, nug.end_char,
                                                     men["start_char"], men["end_char"]):
                            break
                    else:
                        # nug is an incorrect nugget guess
                        for n in nug.document.nuggets:
                            for men in doc["mentions"][attr_name]:
                                if consider_overlap_as_match(n.start_char, n.end_char,
                                                             men["start_char"], men["end_char"]):
                                    # there is a matching nugget in nug's document
                                    print(f"{attr_name}, RETURN OTHER MATCHING NUGGET")
                                    return {
                                        "message": "is-match",
                                        "nugget": n
                                    }
                        else:
                            # there is no matching nugget in nug's document
                            print(f"{attr_name}, NO MATCH IN DOCUMENT")
                            return {
                                "message": "no-match-in-document",
                                "nugget": nug
                            }
                else:
                    # all nuggets are matches
                    print(f"{attr_name}, IS MATCH")
                    return {
                        "message": "is-match",
                        "nugget": nuggets[0]
                    }


            matching_phase(
                document_base,
                automatic_feedback_fn,
                status_fn,
                statistics["matching"]["runs"][str(run)]["statistics"]
            )

            # evaluate the matching process
            for attribute, attribute_name in zip(dataset.ATTRIBUTES, attribute_names):
                results = statistics["matching"]["runs"][str(run)]["results"][attribute]
                results["num_should_be_filled_is_empty"] = 0
                results["num_should_be_filled_is_correct"] = 0
                results["num_should_be_filled_is_incorrect"] = 0
                results["num_should_be_empty_is_empty"] = 0
                results["num_should_be_empty_is_full"] = 0

                for document, aset_document in zip(documents, document_base.documents):
                    found_nuggets = aset_document.attribute_mappings[attribute_name]

                    if document["mentions"][attribute]:  # document states cell's value
                        if found_nuggets == []:
                            results["num_should_be_filled_is_empty"] += 1
                        else:
                            found_nugget = found_nuggets[0]  # TODO: evaluate if there is more than one found nugget
                            for mention in document["mentions"][attribute]:
                                if consider_overlap_as_match(
                                        mention["start_char"],
                                        mention["end_char"],
                                        found_nugget.start_char,
                                        found_nugget.end_char
                                ):
                                    results["num_should_be_filled_is_correct"] += 1
                                    break
                            else:
                                results["num_should_be_filled_is_incorrect"] += 1

                    else:  # document does not state cell's value
                        if found_nuggets == []:
                            results["num_should_be_empty_is_empty"] += 1
                        else:
                            results["num_should_be_empty_is_full"] += 1

                # compute the evaluation metrics
                results["recall"] = \
                    results["num_should_be_filled_is_correct"] / (
                            results["num_should_be_filled_is_correct"]
                            + results["num_should_be_filled_is_incorrect"]
                            + results["num_should_be_filled_is_empty"]
                    )

                if (results["num_should_be_filled_is_correct"] + results["num_should_be_filled_is_incorrect"]) == 0:
                    results["precision"] = 1
                else:
                    results["precision"] = results["num_should_be_filled_is_correct"] / (
                            results["num_should_be_filled_is_correct"]
                            + results["num_should_be_filled_is_incorrect"]
                    )

                if results["precision"] + results["recall"] == 0:
                    results["f1_score"] = 0
                else:
                    results["f1_score"] = 2 * results["precision"] * results["recall"] \
                                          / (results["precision"] + results["recall"])

        # compute the results as the median
        for attribute in dataset.ATTRIBUTES:
            for score in ["recall", "precision", "f1_score"]:
                values = [res["results"][attribute][score] for res in statistics["matching"]["runs"].all_values()]
                statistics["matching"]["results"][attribute][score] = np.median(values)

        ################################################################################################################
        # store the results
        ################################################################################################################
        if not os.path.isdir(f"results/{dataset.NAME}"):
            os.makedirs(f"results/{dataset.NAME}", exist_ok=True)
        with open(f"results/{dataset.NAME}/" + "aset-stanza-ranking" + ".json", "w") as file:
            json.dump(statistics.to_serializable(), file)
