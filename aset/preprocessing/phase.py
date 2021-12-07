import logging
import time
from typing import Any, Dict, List

from aset.config import ConfigurableElement
from aset.data.data import ASETDocumentBase
from aset.preprocessing.embedding import BaseEmbedder
from aset.preprocessing.extraction import BaseExtractor
from aset.preprocessing.normalization import BaseNormalizer
from aset.statistics import Statistics
from aset.status import StatusFunction

logger: logging.Logger = logging.getLogger(__name__)


class PreprocessingPhase(ConfigurableElement):
    """
    Preprocessing phase that applies extractors, normalizers, and embedders on an ASETDocumentBase.

    The preprocessing phase is a configurable element.
    """

    def __init__(
            self, extractors: List[BaseExtractor], normalizers: List[BaseNormalizer], embedders: List[BaseEmbedder]
    ) -> None:
        """
        Initialize the PreprocessingPhase.

        :param extractors: extractors of the preprocessing phase
        :param normalizers: normalizers of the preprocessing phase
        :param embedders: embedders of the preprocessing phase
        """
        super(PreprocessingPhase, self).__init__()
        self._extractors: List[BaseExtractor] = extractors
        self._normalizers: List[BaseNormalizer] = normalizers
        self._embedders: List[BaseEmbedder] = embedders

        logger.debug("Initialized preprocessing phase.")

    def __str__(self) -> str:
        extractors_str: str = "\n".join(f"- {extractor.extractor_str}" for extractor in self._extractors)
        normalizers_str: str = "\n".join(f"- {normalizer.normalizer_str}" for normalizer in self._normalizers)
        embedders_str: str = "\n".join(f"- {embedder.embedder_str}" for embedder in self._embedders)
        return "Extractors:\n{}\n\nNormalizers:\n{}\n\nEmbedders:\n{}".format(
            extractors_str if extractors_str != "" else " -",
            normalizers_str if normalizers_str != "" else " -",
            embedders_str if embedders_str != "" else " -"
        )

    def __eq__(self, other) -> bool:
        return (
                isinstance(other, PreprocessingPhase)
                and self._extractors == other._extractors
                and self._normalizers == other._normalizers
                and self._embedders == other._embedders
        )

    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: StatusFunction,
            statistics: Statistics
    ) -> None:
        """
        Apply the preprocessing phase on the given document base.

        :param document_base: document base to preprocess
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: statistics object to collect statistics
        """
        logger.info(
            f"Execute preprocessing phase on document base with {len(document_base.documents)} documents "
            f"and {len(document_base.attributes)} attributes."
        )
        tick: float = time.time()
        status_fn("Running preprocessing phase...", -1)

        for ix, extractor in enumerate(self._extractors):
            extractor(document_base, status_fn, statistics[f"extractor-{ix}"])

        for ix, normalizer in enumerate(self._normalizers):
            normalizer(document_base, status_fn, statistics[f"normalizer-{ix}"])

        for ix, embedder in enumerate(self._embedders):
            embedder(document_base, status_fn, statistics[f"embedder-{ix}"])

        status_fn("Running preprocessing phase...", 1)
        tack: float = time.time()
        logger.info(f"Executed preprocessing phase in {tack - tick} seconds.")

    @property
    def extractors(self) -> List[BaseExtractor]:
        """Extractors of the preprocessing phase."""
        return self._extractors

    @property
    def normalizers(self) -> List[BaseNormalizer]:
        """Normalizers of the preprocessing phase."""
        return self._normalizers

    @property
    def embedders(self) -> List[BaseEmbedder]:
        """Embedders of the preprocessing phase."""
        return self._embedders

    def to_config(self) -> Dict[str, Any]:
        return {
            "extractors": [extractor.to_config() for extractor in self._extractors],
            "normalizers": [normalizer.to_config() for normalizer in self._normalizers],
            "embedders": [embedder.to_config() for embedder in self._embedders]
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PreprocessingPhase":
        return cls(
            extractors=[BaseExtractor.from_config(extractor_config) for extractor_config in config["extractors"]],
            normalizers=[BaseNormalizer.from_config(normalizer_config) for normalizer_config in config["normalizers"]],
            embedders=[BaseEmbedder.from_config(embedder_config) for embedder_config in config["embedders"]]
        )
