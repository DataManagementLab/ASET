"""
Resource backend for resources that other parts of the system can resort to.

This module implements the singleton pattern for resources that different parts of the system may resort to.
Furthermore, it provides a single method to close all open resources that must always be called before the program
terminates.

The performance-critical properties of the used libraries can be configured here.
"""
import logging
import os

import numpy as np
import stanza
import torch
from sentence_transformers import SentenceTransformer
from stanza import server
from transformers import BertModel, BertTokenizer

logger = logging.getLogger(__name__)

_stanza_tokenize_pipeline = None
_stanza_ner_pipeline = None
_stanford_corenlp_pipeline = None
_sentence_bert = None
_bert_tokenizer_device = None
_fasttext_embedding = None
_glove_embedding = None


def get_stanza_tokenize_pipeline():
    """
    Get the Stanza tokenization pipeline and initialize it if necessary.

    The Stanza tokenization pipeline is a natural language pipeline that tokenizes the given text.
    It is based on the Stanza library: https://stanfordnlp.github.io/stanza/
    Stanza paper: https://arxiv.org/pdf/2003.07082.pdf
    """
    global _stanza_tokenize_pipeline

    # initialize if necessary
    if _stanza_tokenize_pipeline is None:
        logger.info("Initialize the Stanza tokenize pipeline.")
        _stanza_tokenize_pipeline = stanza.Pipeline(
            lang="en",
            processors="tokenize"
        )

    return _stanza_tokenize_pipeline


def get_stanza_ner_pipeline():
    """
    Get the Stanza NER pipeline and initialize it if necessary.

    The Stanza NER pipeline is a natural language pipeline that derives named entities from the given text.
    It is based on the Stanza library: https://stanfordnlp.github.io/stanza/
    Stanza paper: https://arxiv.org/pdf/2003.07082.pdf
    """
    global _stanza_ner_pipeline

    # initialize if necessary
    if _stanza_ner_pipeline is None:
        logger.info("Initialize the Stanza NER pipeline.")
        _stanza_ner_pipeline = stanza.Pipeline(
            lang="en",
            processors="tokenize,ner"
        )

    return _stanza_ner_pipeline


def get_stanford_corenlp_pipeline():
    """
    Get the Stanford Core-NLP pipeline and initialize it if necessary.

    The Stanford-CoreNLP pipeline is a natural language pipeline that derives named entities from the given text.
    It is based on the Stanford-CoreNLP library: https://stanfordnlp.github.io/CoreNLP/
    """
    global _stanford_corenlp_pipeline

    # initialize if necessary
    if _stanford_corenlp_pipeline is None:
        logger.info("Initialize the Stanford Core-NLP pipeline.")
        _stanford_corenlp_pipeline = server.CoreNLPClient(
            annotators=["tokenize", "ssplit", "ner"],
            timeout=240000,
            memory="4G",
            threads=8
        )

    return _stanford_corenlp_pipeline


def get_sentence_bert():
    """
    Get Sentence-BERT and initialize it if necessary.

    Sentence-BERT is a sentence embedding method that embeds natural language text into a high-dimensional vector space.
    Sentence-BERT paper: https://arxiv.org/pdf/1908.10084.pdf
    """
    global _sentence_bert

    # initialize if necessary
    if _sentence_bert is None:
        logger.info("Initialize Sentence-BERT.")
        _sentence_bert = SentenceTransformer(
            "bert-large-nli-mean-tokens"
        )

    return _sentence_bert


def get_bert_tokenizer_device():
    """
    Get the BERT embedding, tokenizer, and device and initialize it if necessary.

    BERT is a contextualized word embedding method that embeds natural language words into a high-dimensional vector
    space.
    We use the transformers library: https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    BERT paper: https://www.aclweb.org/anthology/N19-1423/
    """
    global _bert_tokenizer_device

    # initialize if necessary
    if _bert_tokenizer_device is None:
        logger.info("Initialize BERT.")
        tokenizer = BertTokenizer.from_pretrained(
            "bert-large-cased"
        )
        tokenizer.add_tokens(["[START_MENTION]", "[END_MENTION]"])

        bert = BertModel.from_pretrained(
            "bert-large-cased"
        )
        if torch.cuda.is_available():
            device = torch.device("cuda")
            bert.to(device)
            _bert_tokenizer_device = (bert, tokenizer, device)
        else:
            _bert_tokenizer_device = (bert, tokenizer, None)

    return _bert_tokenizer_device


def get_fasttext_embedding():
    """
    Get the FastText embedding and initialize it if necessary.

    FastText is a word embedding method that embeds natural language words into a high-dimension vector space.
    FastText can be found here: https://fasttext.cc/
    """
    global _fasttext_embedding

    # initialize if necessary
    if _fasttext_embedding is None:
        logger.info("Initialize FastText.")

        path = os.path.join(os.path.dirname(__file__), "..", "..", "wiki-news-300d-1M-subword.vec")
        with open(path, encoding="utf-8", newline="\n", errors="ignore") as file:
            _ = file.readline()  # skip number of words, dimension
            _fasttext_embedding = {}
            n = 0
            for line in file:
                n += 1
                if n == 80000:
                    break
                parts = line.rstrip().split(" ")
                _fasttext_embedding[parts[0]] = np.array([float(part) for part in parts[1:]])

    return _fasttext_embedding


def get_glove_embedding():
    """
    Get the GloVe embedding and initialize it if necessary.

    GloVe is a word embedding method that embeds natural language words into a high-dimensional vector space.
    GloVe can be found here: https://nlp.stanford.edu/projects/glove/
    """
    global _glove_embedding

    # initialize if necessary
    if _glove_embedding is None:
        logger.info("Initialize GloVe.")

        # this is done this way so that this code can be executed from everywhere and still finds the files
        path = os.path.join(os.path.dirname(__file__), "..", "..", "glove.6B.300d.txt")
        with open(path, encoding="utf-8") as f:
            _glove_embedding = {}
            for line in f:
                parts = line.rstrip().split(" ")
                _glove_embedding[parts[0]] = np.array([float(part) for part in parts[1:]])

    return _glove_embedding


def close_all_resources():
    """
    Close all opened resources.

    This method closes all open resources and must always be called before the program terminates.
    """
    global _stanza_tokenize_pipeline
    global _stanza_ner_pipeline
    global _stanford_corenlp_pipeline
    global _sentence_bert
    global _bert_tokenizer_device
    global _fasttext_embedding
    global _glove_embedding

    logger.info("Close all resources.")

    del _stanza_tokenize_pipeline
    _stanza_tokenize_pipeline = None

    del _stanza_ner_pipeline
    _stanza_ner_pipeline = None

    if _stanford_corenlp_pipeline is not None:
        # noinspection PyUnresolvedReferences
        _stanford_corenlp_pipeline.stop()
        del _stanford_corenlp_pipeline
        _stanford_corenlp_pipeline = None

    del _sentence_bert
    _sentence_bert = None

    del _bert_tokenizer_device
    _bert_tokenizer_device = None

    del _fasttext_embedding
    _fasttext_embedding = None

    del _glove_embedding
    _glove_embedding = None

    logger.info("Closed all resources.")
