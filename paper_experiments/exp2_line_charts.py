"""Draw bar charts given result files."""

import logging.config
from glob import glob

from matplotlib import pyplot as plt
import pandas as pd

logging.config.fileConfig("../logging.conf", disable_existing_loggers=False)
logger = logging.getLogger()

if __name__ == "__main__":

    # read the results
    param_values = []
    all_values = []

    folder_path = "../results/aviation/Stanza/max-distance/*.csv"
    file_paths = glob(folder_path)
    for file_path in file_paths:
        param_values.append(float(file_path[file_path.rindex("\\") + 1:-4]))
        all_values.append(pd.read_csv(file_path, header=0, names=["attribute", "recall", "precision", "f1_score", "recall_diff_value", "precision_diff_value", "f1_score_diff_value"]))

    param_values_all_values = list(sorted(list(zip(param_values, all_values)), key=lambda x: x[0]))
    param_values = []
    all_values = []

    for param_value, all_value in param_values_all_values:
        param_values.append(param_value)
        all_values.append(all_value)

    attributes = all_values[0]["attribute"].tolist()

    # draw the diagrams
    for key in ("recall", "precision", "f1_score", "recall_diff_value", "precision_diff_value", "f1_score_diff_value"):
        plt.rcParams["figure.figsize"] = (19, 11)
        plt.rcParams["font.size"] = 32
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['ytick.major.width'] = 2

        markers = ["s", "h", "+", "o", "*"]
        colors = ["#f6b93b", "#e55039", "#4a69bd", "#0a3d62", "#78e08f"]
        highlighted_attributes = [0, 1, 2, 3, 9]
        other_attributes = [4, 6, 7, 8, 10]

        for index in other_attributes:
            attribute = attributes[index]
            values = [df[key][index] for df in all_values]
            x_y = sorted(zip(param_values, values), key=lambda x: x[0])
            plt.plot([v[0] for v in x_y], [v[1] for v in x_y], "x--", ms=4, color="#DBDBDB")

        for index, color, marker in zip(highlighted_attributes, colors, markers):
            attribute = attributes[index]
            values = [df[key][index] for df in all_values]
            x_y = sorted(zip(param_values, values), key=lambda x: x[0])
            plt.plot([v[0] for v in x_y], [v[1] for v in x_y], marker + "--", ms=8, color=color, label=attribute)

        # static
        # file_path = "../results/aviation/Stanza/num-examples/0.csv"
        # static_values = pd.read_csv(file_path, header=0, names=["attribute", "recall", "precision", "f1_score", "recall_diff_value", "precision_diff_value", "f1_score_diff_value"])
        #
        # for index, color, marker in zip(highlighted_attributes, colors, markers):
        #     attribute = attributes[index]
        #     values = [static_values[key][index]]
        #     plt.plot([0], values, marker + "--", ms=18, color=color)
        #
        # for index in other_attributes:
        #     attribute = attributes[index]
        #     values = [static_values[key][index]]
        #     plt.plot([0], values, "x--", ms=8, color="#DBDBDB")

        plt.ylim((-0.05, 1.05))
        # xlabels = [int(xlabel) for xlabel in param_values.copy()]
        # xlabels = [int(xlabel) if xlabel % 5 == 0 or xlabel == 1 else "" for xlabel in [0] + param_values]  # num-interactions
        xlabels = [xlabel if xlabel * 40 % 2 == 0 else "" for xlabel in param_values]  # max-distance
        # plt.xticks([0] + [v + 3 for v in param_values], xlabels)  # num-interactions
        plt.xticks(param_values, xlabels)  # max-distance
        plt.ylabel("F1" if key == "f1_score" else key)
        plt.xlabel("maximum distance threshold")
        plt.legend(loc="lower right")

        plt.subplots_adjust(bottom=0.1, top=0.95, left=0.08, right=0.95)

        plt.savefig(
            folder_path[:-5] + key + ".pdf",
            format="pdf",
            transparent=True
        )
        plt.clf()
