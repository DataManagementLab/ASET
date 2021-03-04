"""Draw bar charts given result files."""

import logging.config

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

logging.config.fileConfig("../logging.conf", disable_existing_loggers=False)
logger = logging.getLogger()

if __name__ == "__main__":

    # read the results
    all_values = []

    paths = [  # FINDME: choose the results file names
        "results/aviation/Stanza/signals/labels-mentions-contexts-positions.csv",
        "results/aviation/Stanza/signals/labels-contexts-positions.csv",
        "results/aviation/Stanza/signals/labels-mentions-positions.csv",
        "results/aviation/Stanza/signals/labels-mentions-contexts.csv"
    ]

    # paths = [  # FINDME: choose the results file names
    #     "results/aviation/Stanza/end2end/aset.csv"
    # ]
    for file_path in paths:
        all_values.append(pd.read_csv(file_path, header=0, names=["attribute", "recall", "precision", "f1_score", "recall_diff_value", "precision_diff_value", "f1_score_diff_value"]))

    attributes = all_values[0]["attribute"].tolist()

    # draw the diagrams
    for key in ("recall", "precision", "f1_score"):
        plt.rcParams["figure.figsize"] = (19, 11)
        plt.rcParams["font.size"] = 32
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['ytick.major.width'] = 2

        fig, ax = plt.subplots()
        x = np.arange(len(attributes))

        labels = ["all signals", "w/o mentions", "w/o contexts", "w/o positions"]
        # labels = ["aset"]
        colors = ["black", "#079992", "#e58e26", "#0a3d62", "#4a69bd", "#78e08f"]
        # colors = ["#0a3d62", "#079992", "#e58e26", "#0a3d62", "#4a69bd", "#78e08f"]  # ["black"]  #
        colors_diff_value = ["white", "white", "white", "white", "white"]  # ["grey"]  #

        for index, (values, color, color_diff_value, label) in enumerate(zip(all_values, colors, colors_diff_value, labels)):
            ax.bar(
                x - 0.4 + index * (0.8 / len(labels)),
                values[key + "_diff_value"],
                color=color_diff_value,
                width=0.8 / len(labels),
                align="edge"
            )

            ax.bar(
                x - 0.4 + index * (0.8 / len(labels)),
                values[key],
                color=color,
                label=label,
                width=0.8 / len(labels),
                align="edge"
            )

        ax.set_ylim((0, 1))
        ax.set_xticks(x)
        labels = ["registration_number" if attribute == "aircraft_registration_number" else attribute for attribute in attributes]
        labels = ["intensive_care" if attribute == "patients_ntensive_care" else attribute for attribute in labels]
        ax.set_xticklabels(labels)
        fig.autofmt_xdate(rotation=45)
        ax.set_ylabel("F1" if key == "f1_score" else key)
        plt.legend(loc="lower right")

        plt.subplots_adjust(bottom=0.33, top=0.95, left=0.1, right=0.95)

        plt.savefig(
            paths[0][:-4] + "-" + key + ".pdf",
            format="pdf",
            transparent=True
        )
        plt.clf()
