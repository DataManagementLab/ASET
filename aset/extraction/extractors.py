"""Extractors that derive extractions from documents."""
import logging
from abc import ABC, abstractmethod

from aset.core.resources import get_stanza_ner_pipeline, get_stanford_corenlp_pipeline
from aset.extraction.common import Document, Extraction

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Facility that processes texts and derives extractions. Super class of all extractors."""

    extractor_str = "BaseExtractor"

    def __eq__(self, other):
        return self.extractor_str == other.extractor_str

    @abstractmethod
    def __call__(self, documents: [Document]):
        """Derive extractions from the given documents."""
        raise NotImplementedError


class StanzaExtractor(BaseExtractor):
    """Extractor built upon the Stanza library's named entity recognizer."""

    extractor_str = "Stanza"

    determine_extraction_type_str = {
        # numbers
        "PERCENT": Extraction.number_extraction_type_str,
        "QUANTITY": Extraction.number_extraction_type_str,
        "ORDINAL": Extraction.number_extraction_type_str,
        "CARDINAL": Extraction.number_extraction_type_str,
        "MONEY": Extraction.number_extraction_type_str,

        # dates and times
        "DATE": Extraction.datetime_extraction_type_str,
        "TIME": Extraction.datetime_extraction_type_str,

        # strings
        "PERSON": Extraction.string_extraction_type_str,
        "NORP": Extraction.string_extraction_type_str,
        "FAC": Extraction.string_extraction_type_str,
        "ORG": Extraction.string_extraction_type_str,
        "GPE": Extraction.string_extraction_type_str,
        "LOC": Extraction.string_extraction_type_str,
        "PRODUCT": Extraction.string_extraction_type_str,
        "EVENT": Extraction.string_extraction_type_str,
        "WORK_OF_ART": Extraction.string_extraction_type_str,
        "LAW": Extraction.string_extraction_type_str,
        "LANGUAGE": Extraction.string_extraction_type_str
    }

    def __init__(self):
        """Load the necessary infrastructure."""
        super(StanzaExtractor, self).__init__()
        # preload the required resources
        get_stanza_ner_pipeline()

    def __str__(self):
        return self.extractor_str

    def __call__(self, documents: [Document]):
        """Derive extractions from the given documents."""

        # run the stanza library
        for i, document in enumerate(documents):
            if i % (len(documents) // 5) == 0:
                logger.info(f"{self.extractor_str}: {round(i / len(documents) * 100)} percent done.")

            stanza_result = get_stanza_ner_pipeline()(document.text)

            # collect the extractions
            first_token_in_sentence_index = 0
            for sentence in stanza_result.sentences:
                for entity in sentence.entities:
                    logger.debug(f"Stanza output:\n{str(entity)}")

                    # determine the type of the extraction as well as the other attributes
                    extraction_type_str = self.determine_extraction_type_str[entity.type]
                    extractor_str = self.extractor_str
                    label = entity.type
                    mention = entity.text
                    position = first_token_in_sentence_index + sentence.tokens.index(entity.tokens[0])
                    context = sentence.text

                    document.extractions.append(
                        Extraction(extraction_type_str, extractor_str, label, mention, context, position)
                    )

                first_token_in_sentence_index += len(sentence.tokens)


class StanfordCoreNLPExtractor(BaseExtractor):
    """Extractor built upon the Stanford CoreNLP library's named entity recognizer."""

    extractor_str = "Stanford-CoreNLP"

    determine_extraction_type_str = {
        # numbers
        "NUMBER": Extraction.number_extraction_type_str,
        "ORDINAL": Extraction.number_extraction_type_str,
        "MONEY": Extraction.number_extraction_type_str,
        "PERCENT": Extraction.number_extraction_type_str,

        # dates and times
        "DATE": Extraction.datetime_extraction_type_str,
        "TIME": Extraction.datetime_extraction_type_str,
        "DURATION": Extraction.datetime_extraction_type_str,
        "SET": Extraction.datetime_extraction_type_str,

        # strings
        "PERSON": Extraction.string_extraction_type_str,
        "LOCATION": Extraction.string_extraction_type_str,
        "ORGANIZATION": Extraction.string_extraction_type_str,
        "MISC": Extraction.string_extraction_type_str,
        "CAUSE_OF_DEATH": Extraction.string_extraction_type_str,
        "CITY": Extraction.string_extraction_type_str,
        "COUNTRY": Extraction.string_extraction_type_str,
        "CRIMINAL_CHARGE": Extraction.string_extraction_type_str,
        "EMAIL": Extraction.string_extraction_type_str,
        "HANDLE": Extraction.string_extraction_type_str,
        "IDEOLOGY": Extraction.string_extraction_type_str,
        "NATIONALITY": Extraction.string_extraction_type_str,
        "RELIGION": Extraction.string_extraction_type_str,
        "STATE_OR_PROVINCE": Extraction.string_extraction_type_str,
        "TITLE": Extraction.string_extraction_type_str,
        "URL": Extraction.string_extraction_type_str,
    }

    def __init__(self):
        """Load the necessary infrastructure."""
        super(StanfordCoreNLPExtractor, self).__init__()
        # preload the required resources
        get_stanford_corenlp_pipeline()

    def __call__(self, documents: [Document]):
        """Derive extractions from the given documents."""

        # run the Stanford CoreNLP library
        for i, document in enumerate(documents):
            if i % (len(documents) // 5) == 0:
                logger.info(f"{self.extractor_str}: {round(i / len(documents) * 100)} percent done.")

            corenlp_result = get_stanford_corenlp_pipeline().annotate(document.text)

            # collect the extractions
            for sentence in corenlp_result.sentence:
                for entity in sentence.mentions:
                    logger.debug(f"Stanford CoreNLP output:\n{str(entity)}")

                    # determine the type of the extraction as well as the other attributes
                    extraction_type_str = self.determine_extraction_type_str[entity.ner]
                    extractor_str = self.extractor_str
                    label = entity.ner
                    mention = entity.entityMentionText
                    position = sentence.tokenOffsetBegin + entity.tokenStartInSentenceInclusive
                    context = " ".join(map(lambda token: token.word, sentence.token))

                    document.extractions.append(
                        Extraction(extraction_type_str, extractor_str, label, mention, context, position)
                    )
