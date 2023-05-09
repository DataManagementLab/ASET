This repository contains the code for our demo publication at SIGMOD'22. We continued the project as [WannaDB](https://link.tuda.systems/wannadb).

# ASET: Ad-hoc Structured Exploration of Text Collections

![ASET Overview](aset-overview.png)

ASET extracts information nuggets from a collection of textual documents and matches them to a list of user-specified
attributes. [Watch our demo video](https://link.tuda.systems/aset-video) or [read our demo paper](https://dl.acm.org/doi/abs/10.1145/3514221.3520174) to learn more about the usage and underlying concepts.

Run `main.py` to start the ASET GUI.

## Data Sets

The evaluation scripts use multiple data sets to evaluate the system.

* The **aviation data set** consists of executive summaries from the NTSB Aviation Accident Reports.
* The **corona data set** consists of summaries of the RKI's daily reports about the Covid-19 situation in Germany.
* The **countries data set** consists of articles about sovereign states from the [T-REx dataset](https://hadyelsahar.github.io/t-rex/)
* The **nobel data set** consists of articles about nobel price winners from the [T-REx dataset](https://hadyelsahar.github.io/t-rex/)
* The **skyscrapers data aset** consists of articles about skyscrapers from the [T-REx dataset](https://hadyelsahar.github.io/t-rex/)

For each data set, there is a `handler` module that is meant to access the data set. The annotated documents of the data
set are stored as `*.json` files in the `documents` folder. The raw documents are stored as `*.txt` files in
the `raw-documents` folder.

## Set up the Project

This project runs with Python 3.9.7.

Check out the `requirements.txt` file to see which packages have to be installed.

You can install them with `pip install -r requirements.txt`.

You may have to install `torch` by hand if you want to use CUDA:

https://pytorch.org/get-started/locally/

Make sure to also install `pytest` to execute the tests.

## Citation

If you use ASET for scientific purposes, please cite it as follows:

```
@inproceedings{haettasch22ASET,
  author = {H\"{a}ttasch, Benjamin and Bodensohn, Jan-Micha and Binnig, Carsten},
  title = {Demonstrating ASET: Ad-Hoc Structured Exploration of Text Collections},
  year = {2022},
  isbn = {9781450392495},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3514221.3520174},
  doi = {10.1145/3514221.3520174},
  abstract = {In this demo, we present ASET, a novel tool to explore the contents of unstructured data (text) by automatically transforming relevant parts into tabular form. ASET works in an ad-hoc manner without the need to curate extraction pipelines for the (unseen) text collection or to annotate large amounts of training data. The main idea is to use a new two-phased approach that first extracts a superset of information nuggets from the texts using existing extractors such as named entity recognizers. In a second step, it leverages embeddings and a novel matching strategy to match the extractions to a structured table definition as requested by the user. This demo features the ASET system with a graphical user interface that allows people without machine learning or programming expertise to explore text collections efficiently. This can be done in a self-directed and flexible manner, and ASET provides an intuitive impression of the result quality.},
  booktitle = {Proceedings of the 2022 International Conference on Management of Data},
  pages = {2393–2396},
  numpages = {4},
  keywords = {interactive text exploration, matching embeddings, text to table},
  location = {Philadelphia, PA, USA},
  series = {SIGMOD '22}
}
```
