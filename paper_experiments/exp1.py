"""
Experiment 1
============
"""

# requires that the match_by_hand has been executed on all documents of the dataset

import logging.config
import traceback

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from aset.core.resources import close_all_resources
from datasets.corona import corona as dataset

logging.config.fileConfig("../logging.conf", disable_existing_loggers=False)
logger = logging.getLogger()

if __name__ == "__main__":
    try:
        extractor_strings = ["Stanford-CoreNLP", "Stanza"]
        documents = dataset.load_dataset()

        # count the results
        nums_mentioned = []
        nums_extracted = {}

        for extractor_str in extractor_strings:
            num_mentioned = []
            num_extracted = []

            for attribute in dataset.ATTRIBUTES:
                mentioned = 0
                extracted = 0

                for document in documents:
                    if document["mentions"][attribute]:
                        mentioned += 1

                    if len(document["evaluation"][extractor_str]["mentions"][attribute]) > 0:
                        extracted += 1

                num_mentioned.append(mentioned)
                num_extracted.append(extracted)

            nums_mentioned = num_mentioned
            nums_extracted[extractor_str] = num_extracted

        # draw the diagram
        plt.rcParams["figure.figsize"] = (19, 11)
        plt.rcParams["font.size"] = 32
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['ytick.major.width'] = 2

        fig, ax = plt.subplots()
        x = np.arange(len(dataset.ATTRIBUTES))
        num_columns = len(extractor_strings) + 1
        column_width = 0.7 / num_columns

        colors = ["#079992", "#e58e26"]
        for i, extractor in enumerate(extractor_strings):
            ax.bar(
                x - 0.35 + i * column_width,
                nums_extracted[extractor],
                width=column_width,
                align="edge",
                label=extractor,
                color=colors[i]
            )

        ax.bar(
            x - 0.35 + (num_columns - 1) * column_width,
            nums_mentioned,
            width=column_width,
            align="edge",
            label="mentioned",
            color="black"
        )

        ax.set_xticks(x)
        labels = dataset.ATTRIBUTES
        labels = ["registration_number" if attribute == "aircraft_registration_number" else attribute for attribute in labels]
        labels = ["intensive_care" if attribute == "patients_intensive_care" else attribute for attribute in labels]
        ax.set_xticklabels(labels)
        fig.autofmt_xdate(rotation=45)
        ax.set_ylabel("number of documents")

        plt.subplots_adjust(bottom=0.33, top=0.95, left=0.1, right=0.95)

        plt.savefig(
            "results/" + dataset.NAME + "/exp0_" + dataset.NAME + ".pdf",
            format="pdf",
            transparent=True
        )

        df = pd.DataFrame(
            columns=["mentioned", "Stanford-CoreNLP", "Stanza"],
            index=labels
        )

        for attribute, mentioned in zip(labels, nums_mentioned):
            df["mentioned"][attribute] = mentioned

        for extractor in extractor_strings:
            for attribute, num_extracted in zip(labels, nums_extracted[extractor]):
                df[extractor][attribute] = num_extracted

        df.to_csv("results/" + dataset.NAME + "/exp0_" + dataset.NAME + ".csv")

        plt.clf()

        close_all_resources()
    except:
        traceback.print_exc()
        close_all_resources()
