import logging
import time
from typing import Dict, Any, List, Optional, Callable

from aset.config import ConfigurableElement
from aset.data.data import ASETDocumentBase
from aset.preprocessing.embedding import BaseEmbedder
from aset.preprocessing.extraction import BaseExtractor
from aset.preprocessing.normalization import BaseNormalizer

logger: logging.Logger = logging.getLogger(__name__)


class PreprocessingPhase(ConfigurableElement):
    """
    Preprocessing phase that applies extractors, normalizers, and embedders on an ASETDocumentBase.

    The preprocessing phase is a configurable element.
    """

    def __init__(
            self,
            extractors: List[BaseExtractor],
            normalizers: List[BaseNormalizer],
            embedders: List[BaseEmbedder]
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

        logger.debug(f"Initialized preprocessing phase.")

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
        return isinstance(other, PreprocessingPhase) and self._extractors == other._extractors and \
               self._normalizers == other._normalizers and self._embedders == other._embedders

    def __call__(
            self,
            document_base: ASETDocumentBase,
            status_fn: Optional[Callable[[str, float], None]] = None,
            statistics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Apply the preprocessing phase on the given document base.

        :param document_base: document base to preprocess
        :param status_fn: callback function to communicate current status (message and progress)
        :param statistics: record to collect statistics
        """
        logger.info(f"Execute preprocessing phase on document base with {len(document_base.documents)} documents "
                    f"and {len(document_base.attributes)} attributes.")

        total_time: float = 0
        if statistics is not None:
            statistics["extractors"] = []
            statistics["normalizers"] = []
            statistics["embedders"] = []

        for ix, extractor in enumerate(self._extractors):
            logger.info(f"Execute extractor [{ix + 1}/{len(self._extractors)}] '{extractor.extractor_str}'.")
            tick: float = time.time()

            if statistics is not None:
                extractor_statistics: Dict[str, Any] = {"extractor_str": extractor.extractor_str}
                if status_fn is not None:
                    status_fn(f"Running {extractor.extractor_str}...", -1)
                extractor(document_base, status_fn, extractor_statistics)
                if status_fn is not None:
                    status_fn(f"Running {extractor.extractor_str}...", 1)
                statistics["extractors"].append(extractor_statistics)
            else:
                if status_fn is not None:
                    status_fn(f"Running {extractor.extractor_str}...", -1)
                extractor(document_base, status_fn)
                if status_fn is not None:
                    status_fn(f"Running {extractor.extractor_str}...", 1)

            tack: float = time.time()
            total_time += tack - tick
            logger.info(f"Executed extractor [{ix + 1}/{len(self._extractors)}] '{extractor.extractor_str}' "
                        f"in {tack - tick} seconds.")

        for ix, normalizer in enumerate(self._normalizers):
            logger.info(f"Execute normalizer [{ix + 1}/{len(self._normalizers)}] '{normalizer.normalizer_str}'.")
            tick: float = time.time()

            if statistics is not None:
                normalizer_statistics: Dict[str, Any] = {"normalizer_str": normalizer.normalizer_str}
                if status_fn is not None:
                    status_fn(f"Running {normalizer.normalizer_str}...", -1)
                normalizer(document_base, status_fn, normalizer_statistics)
                if status_fn is not None:
                    status_fn(f"Running {normalizer.normalizer_str}...", 1)
                statistics["normalizers"].append(normalizer_statistics)
            else:
                if status_fn is not None:
                    status_fn(f"Running {normalizer.normalizer_str}...", -1)
                normalizer(document_base, status_fn)
                if status_fn is not None:
                    status_fn(f"Running {normalizer.normalizer_str}...", 1)

            tack: float = time.time()
            total_time += tack - tick
            logger.info(f"Executed normalizer [{ix + 1}/{len(self._normalizers)}] '{normalizer.normalizer_str}' "
                        f"in {tack - tick} seconds.")

        for ix, embedder in enumerate(self._embedders):
            logger.info(f"Execute embedder [{ix + 1}/{len(self._embedders)}] '{embedder.embedder_str}'.")
            tick: float = time.time()

            if statistics is not None:
                embedder_statistics: Dict[str, Any] = {"embedder_str": embedder.embedder_str}
                if status_fn is not None:
                    status_fn(f"Running {embedder.embedder_str}...", -1)
                embedder(document_base, status_fn, embedder_statistics)
                if status_fn is not None:
                    status_fn(f"Running {embedder.embedder_str}...", 1)
                statistics["embedders"].append(embedder_statistics)
            else:
                if status_fn is not None:
                    status_fn(f"Running {embedder.embedder_str}...", -1)
                embedder(document_base, status_fn)
                if status_fn is not None:
                    status_fn(f"Running {embedder.embedder_str}...", 1)

            tack: float = time.time()
            total_time += tack - tick
            logger.info(f"Executed embedder [{ix + 1}/{len(self._embedders)}] '{embedder.embedder_str}' "
                        f"in {tack - tick} seconds.")

        logger.info(f"Executed preprocessing phase in {total_time} seconds.")

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
