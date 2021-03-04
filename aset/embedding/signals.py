"""
Methods to embed the individual signals.

The embedding methods are implemented as functions. Each function must be called with all extractions from a single
document, since some embedding methods uphold the document context.
"""

import logging

import numpy as np
import torch

from aset.core.resources import get_bert_tokenizer_device, get_sentence_bert, get_fasttext_embedding, \
    get_stanza_tokenize_pipeline, get_glove_embedding

logger = logging.getLogger(__name__)

map_label_to_natural_language = {
    # Stanza extractor
    "QUANTITY": "quantity measurement weight distance",
    "CARDINAL": "cardinal numeral",
    "NORP": "nationality religion political group",
    "FAC": "building airport highway bridge",
    "ORG": "organization",
    "GPE": "country city state",
    "LOC": "location mountain range body of water",
    "PRODUCT": "product vehicle weapon food",
    "EVENT": "event hurricane battle war sports",
    "WORK_OF_ART": "work of art title of book song",
    "LAW": "law document",
    "LANGUAGE": "language",

    # Stanza extractor and Stanford CoreNLP extractor
    "ORDINAL": "ordinal",
    "MONEY": "money",
    "PERCENT": "percentage",
    "DATE": "date period",
    "TIME": "time",
    "PERSON": "person",

    # Stanford CoreNLP extractor
    "DURATION": "duration",
    "SET": "set",
    "NUMBER": "number",
    "LOCATION": "location",
    "ORGANIZATION": "organization",
    "MISC": "misc",
    "CAUSE_OF_DEATH": "cause of death",
    "CITY": "city",
    "COUNTRY": "country",
    "CRIMINAL_CHARGE": "criminal charge",
    "EMAIL": "email",
    "HANDLE": "handle",
    "IDEOLOGY": "ideology",
    "NATIONALITY": "nationality",
    "RELIGION": "religion",
    "STATE_OR_PROVINCE": "state province",
    "TITLE": "title",
    "URL": "url"
}


def context_with_bert_natural_language_embeddings(contexts: [str], mentions: [str]):
    """Embed the given texts with BERT."""
    assert len(contexts) == len(mentions)

    embeddings = []
    for context, mention in zip(contexts, mentions):
        # add mention markers in the context
        start_index = context.index(mention)
        marked_context = "".join([
            context[:start_index],
            " [START_MENTION] ",
            mention,
            " [END_MENTION] ",
            context[start_index + len(mention):]
        ])

        # tokenize the context
        tokenized_context = get_bert_tokenizer_device()[1].encode_plus(marked_context)
        context_tokens = tokenized_context["input_ids"]

        # determine the indices of the mention
        left = context_tokens.index(get_bert_tokenizer_device()[1].get_vocab()["[START_MENTION]"])  # one removed
        right = context_tokens.index(get_bert_tokenizer_device()[1].get_vocab()["[END_MENTION]"]) - 1  # two removed

        # embed the context with markers removed
        input_ids = tokenized_context["input_ids"][:left] \
                    + tokenized_context["input_ids"][left + 1:right + 1] \
                    + tokenized_context["input_ids"][right + 2:]
        token_type_ids = tokenized_context["token_type_ids"][:left] \
                         + tokenized_context["token_type_ids"][left + 1:right + 1] \
                         + tokenized_context["token_type_ids"][right + 2:]
        attention_mask = tokenized_context["attention_mask"][:left] \
                         + tokenized_context["attention_mask"][left + 1:right + 1] \
                         + tokenized_context["attention_mask"][right + 2:]

        input_ids = torch.tensor([input_ids]).to(get_bert_tokenizer_device()[2])
        token_type_ids = torch.tensor([token_type_ids]).to(get_bert_tokenizer_device()[2])
        attention_mask = torch.tensor([attention_mask]).to(get_bert_tokenizer_device()[2])

        outputs = get_bert_tokenizer_device()[0](
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        # compute the average of the output embeddings of the mention tokens
        output_embeddings = outputs[0].cpu()[0].detach().numpy()  # final hidden states
        mention_embeddings = output_embeddings[left:right]
        sum_mention_embeddings = sum(mention_embeddings)
        embeddings.append(np.divide(sum_mention_embeddings, right - left))

        torch.cuda.empty_cache()

    return embeddings


def sbert_natural_language_embeddings(texts: [str]):
    """Embed the given texts with Sentence-BERT."""

    embeddings = get_sentence_bert().encode(texts, show_progress_bar=False)
    torch.cuda.empty_cache()
    return embeddings


def labels_as_sbert_natural_language_embeddings(labels: [str]):
    """Embed the given labels by transforming them to natural language and using Sentence-BERT."""

    labels_in_natural_language = [map_label_to_natural_language.get(label, label) for label in labels]
    return sbert_natural_language_embeddings(labels_in_natural_language)


def fasttext_natural_language_embeddings(texts: [str]):
    """Embed the given texts as averaged FastText token embeddings."""

    embeddings = []
    for text in texts:
        sum_token_embeddings = np.zeros_like(get_fasttext_embedding()["a"])  # default embeddings is all zeros
        num_token_embeddings = 0
        for token in get_stanza_tokenize_pipeline()(text).iter_tokens():
            text = token.text.lower()
            if text in get_fasttext_embedding().keys():
                sum_token_embeddings += get_fasttext_embedding()[text]
                num_token_embeddings += 1
            else:
                logger.error("Out-of-vocabulary token '{}' cannot be embedded by FastText!".format(token))
        if num_token_embeddings == 0:
            logger.error("There are no embeddable tokens in this text: '{}'!".format(text))
            embeddings.append(sum_token_embeddings)
        else:
            embeddings.append(sum_token_embeddings / num_token_embeddings)
    return embeddings


def labels_as_fasttext_natural_language_embeddings(labels: [str]):
    """Embed the given labels by transforming them to natural language and using FastText."""

    labels_in_natural_language = [map_label_to_natural_language.get(label, label) for label in labels]
    return fasttext_natural_language_embeddings(labels_in_natural_language)


def glove_natural_language_embeddings(texts: [str]):
    """Embed the given texts as averaged GloVe token embeddings."""

    embeddings = []
    for text in texts:
        sum_token_embeddings = np.zeros_like(get_glove_embedding()["a"])  # default embeddings is all zeros
        num_token_embeddings = 0
        for token in get_stanza_tokenize_pipeline()(text).iter_tokens():
            text = token.text.lower()
            if text in get_glove_embedding().keys():
                sum_token_embeddings += get_glove_embedding()[text]
                num_token_embeddings += 1
            else:
                logger.error("Out-of-vocabulary token '{}' cannot be embedded by GloVe!".format(token))
        if num_token_embeddings == 0:
            logger.error("There are no embeddable tokens in this text: '{}'!".format(text))
            embeddings.append(sum_token_embeddings)
        else:
            embeddings.append(sum_token_embeddings / num_token_embeddings)
    return embeddings


def labels_as_glove_natural_language_embeddings(labels: [str]):
    """Embed the given labels by transforming them to natural language and using GloVe."""

    labels_in_natural_language = [map_label_to_natural_language.get(label, label) for label in labels]
    return glove_natural_language_embeddings(labels_in_natural_language)


def positions_as_relative_positions(positions: [int]):
    """Embed the given positions relative to the greatest given position."""

    if not positions:
        return []

    greatest_position = max(positions)
    return [position / greatest_position for position in positions]
