"""
The main entry point into Ad-hoc Structured Exploration of Text Collections (ASET).

Run this script to execute ASET in the command line. The documents from the document collection must be stored in the
'input' folder as '*.txt' files. ASET will then ask you to specify the names of the attributes. The output will be
stored in the 'output' folder as a '.csv' file.

ASET extracts information nuggets (extractions) from a collection of documents and matches them to a list of
user-specified attributes. Each document corresponds with a single row in the resulting table.
"""

import csv
import logging.config
import os
import traceback
from glob import glob

from aset.core.resources import close_all_resources
from aset.embedding.aggregation import ExtractionEmbeddingMethod, AttributeEmbeddingMethod
from aset.extraction.common import Document
from aset.extraction.extractionstage import ExtractionStage
from aset.extraction.extractors import StanzaExtractor
from aset.extraction.processors import StanfordCoreNLPDateTimeProcessor, StanfordCoreNLPNumberProcessor, \
    StanfordCoreNLPStringProcessor
from aset.matching.common import Attribute
from aset.matching.matchingstage import MatchingStage
from aset.matching.strategies import TreeSearchExploration, query_user

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger()

# run ASET
if __name__ == "__main__":
    try:
        # prompt the user to specify the attributes and give examples
        print("Enter the names of the attributes. Leave empty and press enter to continue.")
        attributes = []
        while True:
            attribute_name = input("Attribute name: ")
            if attribute_name == "":
                break
            else:
                if attribute_name not in [attribute.label for attribute in attributes]:
                    attributes.append(Attribute(attribute_name))
                else:
                    print("An attribute with this name already exists. Please choose a different name!")

        print("\nEnter example mentions for each attribute. Leave empty and press enter to continue.")
        example_mentions = []
        for attribute in attributes:
            mentions = []
            while True:
                mention = input("Example mention for '{}': ".format(attribute.label))
                if mention == "":
                    break
                else:
                    mentions.append(mention)
            example_mentions.append(mentions)
            print()

        # load the document collection
        logger.info("Load the documents.")
        path = os.path.join(os.path.dirname(__file__), "input", "*.txt")
        file_paths = glob(path)
        documents = []
        for file_path in file_paths:
            with open(file_path, encoding="utf-8") as file:
                documents.append(Document(file.read()))
        logger.info("Loaded {} documents.".format(len(documents)))

        # engage the extraction stage
        extraction_stage = ExtractionStage(
            documents=documents,
            extractors=[
                StanzaExtractor()
            ],
            processors=[
                StanfordCoreNLPDateTimeProcessor(),
                StanfordCoreNLPNumberProcessor(),
                StanfordCoreNLPStringProcessor()
            ],
            embedding_method=ExtractionEmbeddingMethod()
        )
        extraction_stage.derive_extractions()
        extraction_stage.determine_values()
        extraction_stage.compute_extraction_embeddings()

        # engage the matching stage
        matching_stage = MatchingStage(
            documents=extraction_stage.documents,
            attributes=attributes,
            strategy=TreeSearchExploration(
                max_roots=2,
                max_initial_tries=10,
                max_children=2,
                explore_far_factor=1.15,
                max_distance=0.3,
                max_interactions=25
            ),
            embedding_method=AttributeEmbeddingMethod()
        )
        matching_stage.compute_attribute_embeddings()
        matching_stage.incorporate_example_mentions(example_mentions)
        generator = matching_stage.match_extractions_to_attributes()
        document, attribute, extraction, num_interactions = next(generator)
        while True:
            try:
                is_match = query_user(document, attribute, extraction, num_interactions)
                document, attribute, extraction, num_interactions = generator.send((is_match, False))
            except StopIteration:
                break

        # display the results
        print("\n\n\n")
        print(matching_stage.table_str)

        # store the results
        path = os.path.join(os.path.dirname(__file__), "output", "results.csv")
        with open(path, "w", newline="", encoding="utf-8") as file:
            fieldnames = [attribute.label for attribute in matching_stage.attributes]
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=",", quotechar="\"", quoting=csv.QUOTE_ALL)
            for row in matching_stage.rows:
                row_dict = {}
                for attribute_name, extraction in row.extractions.items():
                    if extraction is None:  # no match has been found
                        row_dict[attribute_name] = "[no-match]"
                    elif extraction.value is None:  # no value
                        row_dict[attribute_name] = "[no-value]"
                    else:
                        row_dict[attribute_name] = extraction.value
                writer.writerow(row_dict)

        # close all resources
        close_all_resources()
    except:
        traceback.print_exc()
        close_all_resources()
