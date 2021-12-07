import json
import logging.config
import os

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "results", "aviation", "aset-stanza-ranking.json", )
    with open(path) as file:
        record = json.load(file)

    num_mentions = [record["dataset"]["mentioned"][attribute] for attribute in record["dataset"]["attributes"]]
    num_extracted = [record["preprocessing"]["results"][attribute] for attribute in record["dataset"]["attributes"]]
    percent_extracted = [y / x for x, y in zip(num_mentions, num_extracted)]
    recalls = [record["matching"]["results"][attribute]["recall"] for attribute in record["dataset"]["attributes"]]
    precisions = [
        record["matching"]["results"][attribute]["precision"] for attribute in record["dataset"]["attributes"]
    ]
    f1_scores = [record["matching"]["results"][attribute]["f1_score"] for attribute in record["dataset"]["attributes"]]

    ####################################################################################################################
    # mentions by attribute
    ####################################################################################################################
    _, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=record["dataset"]["attributes"], y=num_mentions, ax=ax)
    ax.set_ylabel("#documents")
    ax.set_title("Number of Documents that Mention the Attribute")
    ax.tick_params(axis="x", labelsize=3)
    plt.savefig(path[:-5] + "-num-mentions.pdf", format="pdf", transparent=True)

    ####################################################################################################################
    # percentage extracted by attribute
    ####################################################################################################################
    _, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=record["dataset"]["attributes"], y=percent_extracted, ax=ax)
    ax.set_ylabel("percentage extracted")
    ax.set_title("Percentage of Extracted Mentions by Attribute")
    ax.tick_params(axis="x", labelsize=3)
    plt.savefig(path[:-5] + "-percent-extracted.pdf", format="pdf", transparent=True)

    ####################################################################################################################
    # F1-Scores by attribute
    ####################################################################################################################
    _, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=record["dataset"]["attributes"], y=f1_scores, ax=ax)
    ax.set_ylabel("F1-Score")
    ax.set_title("E2E F1-Scores by Attribute")
    ax.tick_params(axis="x", labelsize=3)
    plt.savefig(path[:-5] + "-f1-scores.pdf", format="pdf", transparent=True)
