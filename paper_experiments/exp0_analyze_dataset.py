"""Analyze a dataset."""
import traceback

import numpy as np
import pandas as pd

from aset.core.resources import close_all_resources, get_stanza_tokenize_pipeline
from aset.extraction.extractors import StanzaExtractor as Extractor  # FINDME: choose the extractor
from datasets.aviation import aviation as dataset  # FINDME: choose the dataset

if __name__ == "__main__":
    try:
        documents = dataset.load_dataset()

        # lengths of the documents
        lengths = []
        for document in documents:
            lengths.append(len(list(get_stanza_tokenize_pipeline()(document["text"]).iter_tokens())))

        print("Average:", np.average(lengths))
        print("Median:", np.median(lengths))

        # ner types and columns
        ner_types = [
            "PERCENT",
            "QUANTITY",
            "ORDINAL",
            "CARDINAL",
            "MONEY",
            "DATE",
            "TIME",
            "PERSON",
            "NORP",
            "FAC",
            "ORG",
            "GPE",
            "LOC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE"
        ]
        df = pd.DataFrame(columns=dataset.ATTRIBUTES, index=ner_types)
        for attribute in dataset.ATTRIBUTES:
            for ner_type in ner_types:
                df[attribute][ner_type] = 0

        for document in documents:
            for attribute in dataset.ATTRIBUTES:
                indices = document["evaluation"][Extractor.extractor_str]["mentions"][attribute]
                extractions = [document["evaluation"][Extractor.extractor_str]["all_extractions"][i] for i in indices]
                labels = [extraction["label"] for extraction in extractions]
                for label in labels:
                    df[attribute][label] += 1

        df.to_csv("../results/types-labels-matrix.csv")

        close_all_resources()
    except:
        traceback.print_exc()
        close_all_resources()
