"""
Perform the matching step by hand and using rules.

This script is used to enrich the annotated datasets by performing the matching step by hand and using rules. After
applying rules, it prompts the user to create ground truths choosing the correct extractions for every attribute.
"""
import logging.config
import traceback
import os
import difflib

import datasets.aviation.aviation as dataset
from aset.core.resources import close_all_resources
from aset.extraction.common import Document
from aset.extraction.extractionstage import ExtractionStage
from aset.extraction.extractors import StanzaExtractor as Extractor

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger()

if __name__ == "__main__":
    try:
        documents = dataset.load_dataset()
        extractor = Extractor()

        # find an entry that has not been done yet
        document = None
        for d in documents:
            if extractor.extractor_str not in d["evaluation"].keys():
                document = d
                break
        else:
            print("There is no document that has not been done yet!")
            exit()

        evaluation = {
            "mentions": {},
            "mentions_diff_value": {}
        }
        document["evaluation"][extractor.extractor_str] = evaluation

        # derive the extractions using a dummy extraction stage
        aset_document = Document(document["text"])
        extraction_stage = ExtractionStage([aset_document], [extractor], [], None)
        extraction_stage.derive_extractions()
        evaluation["all_extractions"] = [extraction.json_dict for extraction in aset_document.extractions]

        # match the extractions by hand and using rules
        for attribute in dataset.ATTRIBUTES:
            mentions = [mention["mention"] for mention in document["mentions"][attribute]]
            mentions_diff_value = [mention["mention"] for mention in document["mentions_diff_value"][attribute]]

            mentions_indices = []
            mentions_diff_value_indices = []
            left_indices = []

            # match based on rules
            for index, extraction in enumerate(evaluation["all_extractions"]):
                if extraction["mention"] in mentions:
                    mentions_indices.append(index)
                elif extraction["mention"] in mentions_diff_value:
                    mentions_diff_value_indices.append(index)
                else:
                    left_indices.append(index)

            # match by hand
            if document["mentions"][attribute]:
                os.system("cls")
                title = "(mentioned) " + attribute
                print(title)
                print("=" * len(title))

                print("\nfound:")
                for index in mentions_indices:
                    mention = evaluation["all_extractions"][index]["mention"].replace("\n", " ")
                    print("{:4.4} {}".format(str(index), mention))

                print("\n\n\ntype in indices for:\n")
                print("    ".join(mentions))
                print("    ".join("=" * len(mention) for mention in mentions))
                print()

                index_mention_overlap = []
                for index in left_indices:
                    mention = evaluation["all_extractions"][index]["mention"].replace("\n", " ")
                    max_overlap = 0
                    for true_mention in mentions:
                        s = difflib.SequenceMatcher(None, mention, true_mention)
                        _, _, overlap = s.find_longest_match(0, len(mention), 0, len(true_mention))
                        if overlap > max_overlap:
                            max_overlap = overlap
                    mention = mention.replace("\n", " ")
                    index_mention_overlap.append((index, mention, max_overlap))
                index_mention_overlap = sorted(index_mention_overlap, key=lambda x: x[2], reverse=True)
                for index, mention, _ in index_mention_overlap:
                    print("{:4.4} {}".format(str(index), mention))
                print()

                indices = []
                while True:
                    indices = []
                    try:
                        s = input("> ")
                        if s == "close":
                            close_all_resources()
                        parts = s.split()
                        for part in parts:
                            index = int(part)
                            assert index in left_indices
                            indices.append(index)
                        break
                    except:
                        print("Not valid indices!")

                for index in indices:
                    mentions_indices.append(index)
                    left_indices.remove(index)

            if document["mentions_diff_value"][attribute]:
                os.system("cls")
                title = "(different value) " + attribute
                print(title)
                print("=" * len(title))

                print("\nfound:")
                for index in mentions_diff_value_indices:
                    mention = evaluation["all_extractions"][index]["mention"].replace("\n", " ")
                    print("{:4.4} {}".format(str(index), mention))

                print("\n\n\ntype in indices for:\n")
                print("    ".join(mentions_diff_value))
                print("    ".join("=" * len(mention) for mention in mentions_diff_value))
                print()

                index_mention_overlap = []
                for index in left_indices:
                    mention = evaluation["all_extractions"][index]["mention"].replace("\n", " ")
                    max_overlap = 0
                    for true_mention in mentions_diff_value:
                        s = difflib.SequenceMatcher(None, mention, true_mention)
                        _, _, overlap = s.find_longest_match(0, len(mention), 0, len(true_mention))
                        if overlap > max_overlap:
                            max_overlap = overlap
                    mention = mention.replace("\n", " ")
                    index_mention_overlap.append((index, mention, max_overlap))
                index_mention_overlap = sorted(index_mention_overlap, key=lambda x: x[2], reverse=True)
                for index, mention, _ in index_mention_overlap:
                    print("{:4.4} {}".format(str(index), mention))
                print()

                indices = []
                while True:
                    indices = []
                    try:
                        s = input("> ")
                        if s == "close":
                            close_all_resources()
                        parts = s.split()
                        for part in parts:
                            index = int(part)
                            assert index in left_indices
                            indices.append(index)
                        break
                    except:
                        print("Not valid indices!")

                for index in indices:
                    mentions_diff_value_indices.append(index)
                    left_indices.remove(index)

            evaluation["mentions"][attribute] = mentions_indices
            evaluation["mentions_diff_value"][attribute] = mentions_diff_value_indices

            print("\n\n\n\n")

        dataset.write_document(document)

    except:
        traceback.print_exc()
        close_all_resources()
