"""
Access the corona dataset.

The corona dataset consists of the summaries of the RKI's daily situational reports about COVID-19:
https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Situationsberichte/Gesamt.html

The texts have been annotated with information about where they mention the structured values. The evaluation part of
the dataset contains a ground-truth for the matching process for different extractors.

Each entry of the dataset is a json file of the following structure:
{
    "id": "<id of the document>",
    "text": "<summary of the report>",
    "mentions": {
        "<attribute name>": [
            {
            "mention": "<text of the mention>",
            "start": <position of the first character of the mention>,
            "length": <length of the mention>
            }    # for each mention of the attribute in the text
        ]  # for each attribute
    },
    "mentions_diff_value": {
        #  same as "mentions", but with mentions of the same attribute class (e.g. city) but not the desired value
    },
    "evaluation": {
        "<extractor name>": {
                "all_extractions": [
                    <json-serialized extraction>  # for all extractions by the extractor
                ],
                "mentions": {
                    "<attribute name>": [<indices of corresponding extractions>]  # for each attribute
                },
                "mentions_diff_value": {
                    "<attribute name>": [<indices of corresponding extractions>]  # for each attribute
                }
            }  # for some extractors
        }
    }  # for each document
"""

import json
import logging
import os
from glob import glob

logger = logging.getLogger(__name__)

NAME = "corona"

ATTRIBUTES = [
    "date",  # date of the report
    "new_cases",  # number of new cases
    "new_deaths",  # number of new deaths
    "incidence",  # 7-day incidence
    "patients_intensive_care",  # number of people in intensive care
    "vaccinated",  # number of people that have been vaccinated at least once
    "twice_vaccinated"  # number of people that have been vaccinated twice
]


def load_dataset():
    """
    Load the corona dataset.

    This method requires the .txt files in the "datasets/corona/documents/" folder.
    """
    dataset: [] = []

    path = os.path.join(os.path.dirname(__file__), "documents", "*.json")
    for file_path in glob(path):
        with open(file_path, encoding="utf-8") as file:
            dataset.append(json.loads(file.read()))

    return dataset


def write_document(document: {}):
    """
    Write the given document to the dataset.
    """

    path = os.path.join(os.path.dirname(__file__), "documents", document["id"] + ".json")
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(document))
