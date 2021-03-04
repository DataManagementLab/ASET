"""
Access the aviation dataset.

The aviation dataset consists of the executive summaries of the NTSB Aviation Accident Reports:
https://www.ntsb.gov/investigations/AccidentReports/Pages/aviation.aspx

The texts have been annotated with information about where they mention the structured values. The evaluation part of
the dataset contains a ground-truth for the matching process for different extractors.

Each entry of the dataset is a json file of the following structure:
{
    "id": "<id of the document>",
    "text": "<executive summary of the report>",
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

NAME = "aviation"

ATTRIBUTES = [
    "event_date",  # date of the event
    "location_city",  # city or place closest to the site of the event
    "location_state",  # state the city is located in
    "airport_code",  # code of the airport
    "airport_name",  # airport name
    "aircraft_damage",  # severity of the damage to the aircraft
    "aircraft_registration_number",  # registration number of the involved aircraft
    "aircraft_make",  # name of the aircraft's manufacturer
    "aircraft_model",  # alphanumeric aircraft model code
    "far_description",  # applicable regulation part or authority
    "air_carrier",  # name of the operator of the aircraft
    "weather_condition"  # weather conditions at the time of the event
]


def load_dataset():
    """
    Load the aviation dataset.

    This method requires the .txt files in the "datasets/aviation/documents/" folder.
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
