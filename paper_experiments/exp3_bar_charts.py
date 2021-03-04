"""Draw bar charts given result files."""

import logging.config

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

logging.config.fileConfig("../logging.conf", disable_existing_loggers=False)
logger = logging.getLogger()

if __name__ == "__main__":

    # read the results
    path = "../results/aviation/Stanza/end2end/aset.csv"
    values = pd.read_csv(path, header=0, names=["attribute", "recall", "precision", "f1_score", "recall_diff_value", "precision_diff_value", "f1_score_diff_value"])

    attributes = values["attribute"].tolist()

    # draw the diagrams
    plt.rcParams["figure.figsize"] = (19, 11)
    plt.rcParams["font.size"] = 32
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['ytick.major.width'] = 2

    fig, ax = plt.subplots()
    x = np.arange(5, 5 + len(attributes))

    # ASET f1-scores by attribute
    ax.bar(
        x,
        values["f1-score"],
        color="#e58e26",  # "black"
        width=0.7
    )

    for x_value, value in zip(x, values["f1-score"]):
        ax.text(
                    x_value,
                    value + 0.01,
                    str(round(value, 2)),
                    fontsize=20,
                    horizontalalignment="center"
                )

    # COMA avg. recall  (aviation: 0.096, corona: 1/6)
    ax.bar(
        [0],
        [1/6],
        color="#079992",  # "grey"
        width=0.7
    )
    ax.text(
        0,
        1/6 + 0.01,
        str(round(1/6, 2)),
        fontsize=20,
        horizontalalignment="center"
    )

    # ASET avg. recall
    ax.bar(
        [1],
        [np.mean(values["recall"])],
        color="#e58e26",  # "black"
        width=0.7
    )
    ax.text(
        1,
        np.mean(values["recall"]) + 0.01,
        str(round(np.mean(values["recall"]), 2)),
        fontsize=20,
        horizontalalignment="center"
    )

    # ASET avg. f1-score
    ax.bar(
        [3],
        [np.mean(values["f1_score"])],
        color="#e58e26",  # "black"
        width=0.7
    )
    ax.text(
        3,
        np.mean(values["f1_score"]) + 0.01,
        str(round(np.mean(values["f1_score"]), 2)),
        fontsize=20,
        horizontalalignment="center"
    )

    ax.set_ylim((0, 1))
    ax.set_xticks([0, 1, 3] + list(x))
    labels = ["registration_number" if attribute == "aircraft_registration_number" else attribute for attribute in attributes]
    labels = ["intensive_care" if attribute == "patients_intensive_care" else attribute for attribute in labels]
    ax.tick_params(labelsize=26)
    ax.set_xticklabels(["Ø COMA Recall", "Ø ASET Recall", "Ø ASET F1"] + labels)
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel("Recall")
    ax2 = ax.twinx()
    ax2.set_ylabel("F1")

    plt.subplots_adjust(bottom=0.3, top=0.95, left=0.1, right=0.9)

    plt.savefig(
        path[:-4] + ".pdf",
        format="pdf",
        transparent=True
    )
    plt.clf()
