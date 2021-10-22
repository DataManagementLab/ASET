import abc
import logging
import os
import time
from typing import Optional, Dict, Any, Type, List, Union

import numpy as np
import spacy
import stanza
import torch
from sentence_transformers import SentenceTransformer
from spacy import Language
from stanza import Pipeline
from transformers import BertTokenizer, BertModel, BertTokenizerFast

logger: logging.Logger = logging.getLogger(__name__)

RESOURCES: Dict[str, Type["BaseResource"]] = {}


def register_resource(resource: Type["BaseResource"]) -> Type["BaseResource"]:
    """Register the given resource."""
    RESOURCES[resource.resource_str] = resource
    return resource


class BaseResource(abc.ABC):
    """
    Resource that may be used by other elements of the system.

    Resources are capabilities that may be used by other elements of the system. Each resource is a class that describes
    how the resource may be loaded ('load'), accessed ('resource'), and unloaded ('unload'). Each resource comes with an
    identifier ('resource_str'). Resources are managed by the resource manager, which can load the same resource once
    and provide it to multiple users and also manages to close all resources when the program ends.
    """
    resource_str: str = "BaseResource"

    @classmethod
    @abc.abstractmethod
    def load(cls) -> "BaseResource":
        """Load the resource."""
        raise NotImplementedError

    @abc.abstractmethod
    def unload(self) -> None:
        """Unload the resource."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def resource(self) -> Any:
        """Access the resource."""
        raise NotImplementedError


@register_resource
class StanzaNERPipeline(BaseResource):
    """
    Stanza pipeline for named entity recognition.

    See https://stanfordnlp.github.io/stanza/
    """
    resource_str: str = "StanzaNERPipeline"

    def __init__(self) -> None:
        """Initialize the StanzaNERPipeline."""
        super(StanzaNERPipeline, self).__init__()
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "stanza")
        self._stanza_ner_pipeline: Pipeline = Pipeline(
            lang="en",
            processors="tokenize,mwt,pos,ner",
            model_dir=path,
            verbose=False
        )

    @classmethod
    def load(cls) -> "StanzaNERPipeline":
        # download the stanza resources if necessary
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "stanza")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "stanza", "en")
        if not os.path.isdir(path):
            path: str = os.path.join(os.path.dirname(__file__), "..", "models", "stanza")
            logger.info("Download the stanza 'en' language package.")
            stanza.download("en", path)
        return cls()

    def unload(self) -> None:
        del self._stanza_ner_pipeline

    @property
    def resource(self) -> Pipeline:
        return self._stanza_ner_pipeline


class BaseFastTextEmbedding(BaseResource, abc.ABC):
    """
    Base class for all FastText embeddings.

    See https://fasttext.cc/
    """
    resource_str: str = "BaseFastTextEmbedding"
    _num_vectors: Optional[int] = None

    def __init__(self) -> None:
        """Initialize the FastText embedding."""
        super(BaseFastTextEmbedding, self).__init__()
        self._fast_text_embedding: Dict[str, np.array] = {}
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "fasttext", "wiki-news-300d-1M-subword.vec")
        with open(path, "r", encoding="utf-8", newline="\n", errors="ignore") as file:
            _ = file.readline()  # skip number of words, dimension
            n: int = 0
            for line in file:
                if self._num_vectors is not None and n >= self._num_vectors:
                    break
                n += 1
                parts: List[str] = line.rstrip().split(" ")
                self._fast_text_embedding[parts[0]] = np.array([float(part) for part in parts[1:]])

    @classmethod
    def load(cls) -> "BaseFastTextEmbedding":
        # check that the FastText model has been downloaded
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "fasttext")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "fasttext", "wiki-news-300d-1M-subword.vec")
        if not os.path.isfile(path):
            logger.error("You have to download the model by hand and place it in the appropriate folder!")
            assert False, "You have to download the model by hand and place it in the appropriate folder!"
        return cls()

    def unload(self) -> None:
        del self._fast_text_embedding

    @property
    def resource(self) -> Dict[str, np.array]:
        return self._fast_text_embedding


@register_resource
class FastTextEmbedding100000(BaseFastTextEmbedding):
    """FastText embedding that includes only the 100000 first vectors."""
    resource_str: str = "FastTextEmbedding100000"
    _num_vectors: Optional[int] = 100000


@register_resource
class FastTextEmbedding(BaseFastTextEmbedding):
    """FastText embedding that includes all vectors."""
    resource_str: str = "FastTextEmbedding"
    _num_vectors: Optional[int] = None


class BaseSpacyResource(BaseResource, abc.ABC):
    """
    Base class for all spacy-based resources.

    See https://spacy.io/
    """
    resource_str: str = "BaseSpacyResource"
    _spacy_package_str: str = "BaseSpacyPackageStr"

    def __init__(self) -> None:
        """Initialize the spacy model."""
        super(BaseSpacyResource, self).__init__()
        self._spacy_nlp: Language = spacy.load(self._spacy_package_str)

    @classmethod
    def load(cls) -> "BaseSpacyResource":
        # download the spacy model if necessary
        if not spacy.util.is_package(cls._spacy_package_str):
            logger.info(f"Download the spacy package '{cls._spacy_package_str}'.")
            spacy.cli.download(cls._spacy_package_str)
            logger.error("Interpreter must be restarted after installing spacy package.")
            assert False, "Interpreter must be restarted after installing spacy package."

        return cls()

    def unload(self) -> None:
        del self._spacy_nlp

    @property
    def resource(self) -> Language:
        return self._spacy_nlp


@register_resource
class SpacyEnCoreWebTrf(BaseSpacyResource):
    """Spacy 'en_core_web_trf' model."""
    resource_str: str = "SpacyEnCoreWebTrf"
    _spacy_package_str: str = "en_core_web_trf"


@register_resource
class SpacyEnCoreWebLg(BaseSpacyResource):
    """Spacy 'en_core_web_lg' model."""
    resource_str: str = "SpacyEnCoreWebLg"
    _spacy_package_str: str = "en_core_web_lg"


@register_resource
class SpacyEnCoreWebMd(BaseSpacyResource):
    """Spacy 'en_core_web_md' model."""
    resource_str: str = "SpacyEnCoreWebMd"
    _spacy_package_str: str = "en_core_web_md"


@register_resource
class SpacyEnCoreWebSm(BaseSpacyResource):
    """Spacy 'en_core_web_sm' model."""
    resource_str: str = "SpacyEnCoreWebSm"
    _spacy_package_str: str = "en_core_web_sm"


@register_resource
class SpacyEnCoreSciMd(BaseResource):
    """
    Spacy 'en_core_sci_md' model.

    See https://allenai.github.io/scispacy/
    """
    resource_str: str = "SpacyEnCoreSciMd"

    def __init__(self) -> None:
        """Initialize the spacy model."""
        super(BaseResource, self).__init__()
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "spacy", "en_core_sci_md-0.4.0",
                                 "en_core_sci_md", "en_core_sci_md-0.4.0")
        self._spacy_nlp: Language = spacy.load(path)

    @classmethod
    def load(cls) -> "SpacyEnCoreSciMd":
        # check that the spacy model has been downloaded
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "spacy")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "spacy", "en_core_sci_md-0.4.0")
        if not os.path.isdir(path):
            logger.error("You have to download the model by hand and place it in the appropriate folder!")
            assert False, "You have to download the model by hand and place it in the appropriate folder!"
        return cls()

    def unload(self) -> None:
        del self._spacy_nlp

    @property
    def resource(self) -> Language:
        return self._spacy_nlp


@register_resource
class SpacyEnNerCraftMd(BaseResource):
    """
    Spacy 'en_ner_craft_md' model.

    See https://allenai.github.io/scispacy/
    """
    resource_str: str = "SpacyEnNerCraftMd"

    def __init__(self) -> None:
        """Initialize the spacy model."""
        super(BaseResource, self).__init__()
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "spacy", "en_ner_craft_md-0.4.0",
                                 "en_ner_craft_md", "en_ner_craft_md-0.4.0")
        self._spacy_nlp: Language = spacy.load(path)

    @classmethod
    def load(cls) -> "SpacyEnNerCraftMd":
        # check that the spacy model has been downloaded
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "spacy")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            path: str = os.path.join(os.path.dirname(__file__), "..", "models", "spacy", "en_ner_craft_md-0.4.0")
        if not os.path.isdir(path):
            logger.error("You have to download the spacy model by hand and place it in the appropriate folder!")
            assert False, "You have to download the spacy model by hand and place it in the appropriate folder!"
        return cls()

    def unload(self) -> None:
        del self._spacy_nlp

    @property
    def resource(self) -> Language:
        return self._spacy_nlp


class BaseBERTResource(BaseResource):
    """
    Base class for all BERT-based resources.

    See https://huggingface.co/transformers/model_doc/bert.html
    """
    resource_str: str = "BaseBERTResource"
    _bert_model_str: str = "BaseBertModelStr"

    def __init__(self) -> None:
        """Initialize the BERT resource."""
        super(BaseBERTResource, self).__init__()
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "transformers")

        self._tokenizer: BertTokenizer = BertTokenizerFast.from_pretrained(
            self._bert_model_str,
            cache_dir=path
        )
        self._tokenizer.add_tokens(["[START_MENTION]", "[END_MENTION]", "[MASK]"])

        self._model: BertModel = BertModel.from_pretrained(
            self._bert_model_str,
            cache_dir=path
        )

        if torch.cuda.is_available():
            self._device: Optional[Any] = torch.device("cuda")
            self._model.to(self._device)
        else:
            self._device: Optional[Any] = None

    @classmethod
    def load(cls) -> "BaseBERTResource":
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "transformers")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        return cls()

    def unload(self) -> None:
        del self._tokenizer
        del self._model
        del self._device

    @property
    def resource(self) -> Dict[str, Any]:
        return {
            "tokenizer": self._tokenizer,
            "model": self._model,
            "device": self._device
        }


@register_resource
class BertLargeCasedResource(BaseBERTResource):
    """BERT 'bert-large-cased' model."""
    resource_str: str = "BertLargeCasedResource"
    _bert_model_str: str = "bert-large-cased"


class BaseSBERTResource(BaseResource, abc.ABC):
    """
    Base class for all SBERT-based resources.

    See https://sbert.net/
    """
    resource_str: str = "BaseSBERTResource"
    _sbert_model_str: str = "BaseSBERTModelStr"

    def __init__(self) -> None:
        """Initialize the SBERT resource."""
        super(BaseSBERTResource, self).__init__()

        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "sentence-transformers")
        self._sbert_model: SentenceTransformer = SentenceTransformer(
            self._sbert_model_str,
            cache_folder=path
        )

    @classmethod
    def load(cls) -> "BaseSBERTResource":
        path: str = os.path.join(os.path.dirname(__file__), "..", "models", "sentence-transformers")
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        return cls()

    def unload(self) -> None:
        del self._sbert_model

    @property
    def resource(self) -> SentenceTransformer:
        return self._sbert_model


@register_resource
class SBERTBertLargeNliMeanTokensResource(BaseSBERTResource):
    """SBERT 'bert-large-nli-mean-tokens' model."""
    resource_str: str = "SBERTBertLargeNliMeanTokensResource"
    _sbert_model_str: str = "bert-large-nli-mean-tokens"


MANAGER: Optional["ResourceManager"] = None


class ResourceManager:
    """
    The resource manager provides the system with access to the resources.

    It loads the resources when they are requested and ensures that they are closed when the program finishes. The
    resource manager implements the singleton pattern, i.e. there should always be at most one resource manager. The
    resource manager should always be accessed using the resources.MANAGER module variable. To set up the resource
    manager in a program, use it as a Python context manager to make sure that all resources are closed when the
    program finishes.
    """

    def __init__(self) -> None:
        """Initialize the resource manager."""
        global MANAGER

        # check that this is the only resource manager
        if MANAGER is not None:
            logger.error("There can only be one resource manager!")
            assert False, "There can only be one resource manager!"
        else:
            MANAGER = self

        self._resources: Dict[str, BaseResource] = {}

        logger.info("Initialized the resource manager.")

    def __enter__(self) -> "ResourceManager":
        """
        Enter the resource manager context.

        :return: the resource manager itself
        """
        logger.info("Entered the resource manager.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the resource manager context."""
        logger.info("Unload all resources.")
        tick: float = time.time()

        # close resources
        for resource_str in list(self._resources.keys()):
            self.unload(resource_str)

        tack: float = time.time()
        logger.info(f"Unloaded all resources in {tack - tick} seconds.")
        logger.info("Exited the resource manager.")

    def __str__(self) -> str:
        resources_str: str = "\n".join(f"- {resource_str}" for resource_str in self._resources.keys())
        return "Currently loaded resources:\n{}".format(
            resources_str if resources_str != "" else " -"
        )

    def load(self, resource: Union[str, Type[BaseResource]]) -> None:
        """
        Load a resource.

        :param resource: resource class or identifier of the resource to load
        """
        if type(resource) == str:
            resource_str: str = resource
        else:
            resource_str: str = resource.resource_str

        if resource_str not in RESOURCES.keys():
            logger.error(f"Unknown resource '{resource_str}'!")
            assert False, f"Unknown resource '{resource_str}'!"
        elif resource_str in self._resources:
            logger.info(f"Resource '{resource_str}' already loaded.")
        else:
            logger.info(f"Load resource '{resource_str}'.")
            tick: float = time.time()
            self._resources[resource_str] = RESOURCES[resource_str].load()
            tack: float = time.time()
            logger.info(f"Loaded resource '{resource_str}' in {tack - tick} seconds.")

    def unload(self, resource: Union[str, Type[BaseResource]]) -> None:
        """
        Unload a resource.

        :param resource: resource class or identifier of the resource to load
        """
        if type(resource) == str:
            resource_str: str = resource
        else:
            resource_str: str = resource.resource_str

        if resource_str not in RESOURCES.keys():
            logger.error(f"Unknown resource '{resource_str}'!")
            assert False, f"Unknown resource '{resource_str}'!"
        elif resource_str not in self._resources:
            logger.error(f"Resource '{resource_str}' is not loaded!")
            assert False, f"Resource '{resource_str}' is not loaded!"
        else:
            logger.info(f"Unload resource '{resource_str}'.")
            tick: float = time.time()
            self._resources[resource_str].unload()
            del self._resources[resource_str]
            tack: float = time.time()
            logger.info(f"Unloaded resource '{resource_str}' in {tack - tick} seconds.")

    def __getitem__(self, resource: Union[str, Type[BaseResource]]) -> Any:
        """
        Access a resource.

        :param resource: resource class or identifier of the resource to load
        :return: the resource
        """
        if type(resource) == str:
            resource_str: str = resource
        else:
            resource_str: str = resource.resource_str

        if resource_str not in RESOURCES.keys():
            logger.error(f"Unknown resource '{resource_str}'!")
            assert False, f"Unknown resource '{resource_str}'!"
        elif resource_str not in self._resources:
            logger.error(f"Resource '{resource_str}' is not loaded!")
            assert False, f"Resource '{resource_str}' is not loaded!"
        else:
            return self._resources[resource_str].resource
