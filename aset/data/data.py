import functools
import logging
import time
from typing import List, Dict, Any, Optional, Union, Type

import bson

from aset.data.annotations import BaseAnnotation
from aset.data.signals import BaseSignal

logger: logging.Logger = logging.getLogger(__name__)


class ASETNugget:
    """
    Information piece obtained from a document.

    The information piece corresponds with a span of text in the document (its provenance). Since the ASETNugget does
    not store the actual text but rather the indices of the span in the document, it only functions together with the
    document.
    """

    def __init__(
            self,
            document: "ASETDocument",
            start_char: int, end_char: int,
            extractor_str: Optional[str],
            type_str: Optional[str],
            value: Optional[str]
    ) -> None:
        """
        Initialize the ASETNugget.

        :param document: document from which it has been obtained
        :param start_char: index of the first character of the span (inclusive)
        :param end_char: index of the first character after the span (exclusive)
        :param extractor_str: identifier of the extractor that obtained this nugget
        :param type_str: identifier of the nugget's data type
        :param value: structured value of the nugget
        """
        self._document: "ASETDocument" = document
        self._start_char: int = start_char
        self._end_char: int = end_char

        self._extractor_str: Optional[str] = extractor_str
        self._type_str: Optional[str] = type_str
        self._value: Optional[str] = value

        self._signals: Dict[str, BaseSignal] = {}

    def __str__(self) -> str:
        return f"'{self.text}'"

    def __repr__(self) -> str:
        return f"ASETNugget({repr(self._document)}, {self._start_char}, {self._end_char})"

    def __hash__(self) -> int:
        # note that two nuggets referring to the same span will always have the same hash value
        return hash((self._document, self._start_char, self._end_char))

    def __eq__(self, other) -> bool:
        return isinstance(other, ASETNugget) and self._document.name == other._document.name and \
               self._start_char == other._start_char and self._end_char == other._end_char and \
               self.extractor_str == other.extractor_str and self._type_str == other._type_str and \
               self._value == other._value and self._signals == other._signals

    @property
    def document(self) -> "ASETDocument":
        """Document from which the nugget has been derived."""
        return self._document

    @property
    def start_char(self) -> int:
        """Index of the first character of the span (inclusive)."""
        return self._start_char

    @property
    def end_char(self) -> int:
        """Index of the first character after the span (exclusive)."""
        return self._end_char

    @functools.cached_property
    def text(self) -> str:
        """Actual text of the span."""
        return self._document.text[self._start_char:self._end_char]

    @property
    def extractor_str(self) -> Optional[str]:
        """Identifier of the extractor that obtained this nugget."""
        return self._extractor_str

    @property
    def type_str(self) -> Optional[str]:
        """Identifier of the nugget's data type."""
        return self._type_str

    @property
    def value(self) -> Optional[str]:
        """Structured value of the nugget."""
        return self._value

    @property
    def signals(self) -> Dict[str, BaseSignal]:
        """Signal values associated with the nugget."""
        return self._signals

    def __getitem__(self, item: Union[str, Type[BaseSignal]]) -> Any:
        if type(item) == str:
            signal_str: str = item
        else:
            signal_str: str = item.signal_str

        if signal_str not in self._signals.keys():
            assert False, f"No value for signal '{signal_str}'!"
        return self._signals[signal_str].value

    def __setitem__(self, key: Union[str, Type[BaseSignal]], value: Union[BaseSignal, Any]):
        if type(key) == str:
            signal_str: str = key
        else:
            signal_str: str = key.signal_str

        if signal_str not in self._signals.keys():
            if not isinstance(value, BaseSignal):
                assert False, f"Signal '{signal_str}' has not been initialized yet!"
            self._signals[signal_str] = value
        if isinstance(value, BaseSignal):
            self._signals[signal_str] = value
        else:
            self._signals[signal_str].value = value


class ASETAttribute:
    """
    Attribute that is populated with information from the documents.

    An ASETAttribute is a class of information that is obtained from the documents. Each ASETDocument may store mappings
    that populate the attribute with ASETNuggets from the document.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the ASETAttribute.

        :param name: name of the attribute (must be unique)
        """
        self._name: str = name

        self._signals: Dict[str, BaseSignal] = {}

    def __str__(self) -> str:
        return f"'{self._name}'"

    def __repr__(self) -> str:
        return f"ASETAttribute('{self._name}')"

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other) -> bool:
        return isinstance(other, ASETAttribute) and self._name == other._name and self._signals == other._signals

    @property
    def name(self) -> str:
        """Name of the attribute."""
        return self._name

    @property
    def signals(self) -> Dict[str, BaseSignal]:
        """Signal values associated with the attribute."""
        return self._signals

    def __getitem__(self, item: Union[str, Type[BaseSignal]]) -> Any:
        if type(item) == str:
            signal_str: str = item
        else:
            signal_str: str = item.signal_str

        if signal_str not in self._signals.keys():
            assert False, f"No value for signal '{signal_str}'!"
        return self._signals[signal_str].value

    def __setitem__(self, key: Union[str, Type[BaseSignal]], value: Union[BaseSignal, Any]):
        if type(key) == str:
            signal_str: str = key
        else:
            signal_str: str = key.signal_str

        if signal_str not in self._signals.keys():
            if not isinstance(value, BaseSignal):
                assert False, f"Signal '{signal_str}' has not been initialized yet!"
            self._signals[signal_str] = value
        if isinstance(value, BaseSignal):
            self._signals[signal_str] = value
        else:
            self._signals[signal_str].value = value


class ASETDocument:
    """
    Textual document.

    The ASETDocument actually owns the text of the document it represents. It stores a list of all the nuggets derived
    from it. Furthermore, it stores mappings from attribute names to lists of ASETNuggets derived from the document.
    """

    def __init__(self, name: str, text: str) -> None:
        """
        Initialize the ASETDocument.

        :param name: name of the document (must be unique)
        :param text: text of the document
        """
        self._name: str = name
        self._text: str = text

        self._nuggets: List[ASETNugget] = []
        self._attribute_mappings: Dict[str, List[ASETNugget]] = {}

        self._annotations: Dict[str, BaseAnnotation] = {}

    def __str__(self) -> str:
        return f"'{self._text}'"

    def __repr__(self) -> str:
        return f"ASETDocument('{self._name}', '{self._text}')"

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other) -> bool:
        return isinstance(other, ASETDocument) and self._name == other._name and self._text == other._text \
               and self._nuggets == other._nuggets and self._attribute_mappings == other._attribute_mappings \
               and self._annotations == other._annotations

    @property
    def name(self) -> str:
        """Name of the document."""
        return self._name

    @property
    def text(self) -> str:
        """Text of the document."""
        return self._text

    @property
    def nuggets(self) -> List[ASETNugget]:
        """Nuggets obtained from the document."""
        return self._nuggets

    @property
    def attribute_mappings(self) -> Dict[str, List[ASETNugget]]:
        """Mappings between attribute names and lists of nuggets associated with them."""
        return self._attribute_mappings

    @property
    def annotations(self) -> Dict[str, BaseAnnotation]:
        """Annotation values associated with the document."""
        return self._annotations

    def __getitem__(self, item: Union[str, Type[BaseAnnotation]]) -> Any:
        if type(item) == str:
            annotation_str: str = item
        else:
            annotation_str: str = item.annotation_str

        if annotation_str not in self._annotations.keys():
            assert False, f"No value for annotation '{annotation_str}'!"
        return self._annotations[annotation_str].value

    def __setitem__(self, key: Union[str, Type[BaseAnnotation]], value: Union[BaseAnnotation, Any]):
        if type(key) == str:
            annotation_str: str = key
        else:
            annotation_str: str = key.annotation_str

        if annotation_str not in self._annotations.keys():
            if not isinstance(value, BaseAnnotation):
                assert False, f"Annotation '{annotation_str}' has not been initialized yet!"
            self._annotations[annotation_str] = value
        if isinstance(value, BaseAnnotation):
            self._annotations[annotation_str] = value
        else:
            self._annotations[annotation_str].value = value


class ASETDocumentBase:
    """
    Collection of documents that provides information.

    The ASETDocumentBase manages the documents and attributes. Furthermore, it provides the serialization capabilities
    and the means to validate its consistency.
    """

    def __init__(self, documents: List[ASETDocument], attributes: List[ASETAttribute]) -> None:
        """
        Initialize the ASETDocumentBase.

        :param documents: documents of the document base
        :param attributes: attributes of the document base
        """
        self._documents: List[ASETDocument] = documents
        self._attributes: List[ASETAttribute] = attributes

    def __str__(self) -> str:
        return f"({len(self._documents)} documents, {len(self.nuggets)} nuggets, {len(self._attributes)} attributes)"

    def __repr__(self) -> str:
        return "ASETDocumentBase([{}], [{}])".format(
            ", ".join(repr(document) for document in self._documents),
            ", ".join(repr(attribute) for attribute in self._attributes)
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, ASETDocumentBase) and self._documents == other._documents \
               and self._attributes == other._attributes

    @property
    def documents(self) -> List[ASETDocument]:
        """Documents of the document base."""
        return self._documents

    @property
    def attributes(self) -> List[ASETAttribute]:
        """Attributes of the document base."""
        return self._attributes

    @property
    def nuggets(self) -> List[ASETNugget]:
        """All nuggets of the document base."""
        nuggets: List[ASETNugget] = []
        for document in self._documents:
            nuggets += document.nuggets
        return nuggets

    def to_table_dict(
            self,
            kind: Optional[str] = None
    ) -> Dict[str, List[Union[str, Optional[List[Union[ASETNugget, Optional[str]]]]]]]:
        """
        Table representation of the information nuggets in the document base.

        The table is stored as a dictionary of columns. It uses the names of attributes and documents, and the parameter
        'kind' determines whether the nuggets *text*, *value* or the nuggets themselves should be stored. A cell can
        either be None, in which case the ASETDocument does not know of the ASETAttribute, or a (possibly empty) list of
        texts/values/nuggets.

        =================================================================
        | 'document-name' | 'attribute-1'        | 'attribute-2'        |
        =================================================================
        | 'document-1'    | [nugget-1, nugget-2] | []                   |
        | 'document-2'    | None                 | [nugget-5]           |
        | 'document-3'    | [nugget-4]           | [nugget-6, nugget-7] |
        =================================================================

        :param kind: whether the nuggets *text*, *value*, or the nuggets themselves (*None*) should be stored.
        :return: table representation of the information nuggets in the document base
        """
        result: Dict[str, List[Union[str, Optional[List[Union[ASETNugget, Optional[str]]]]]]] = {
            "document-name": [document.name for document in self._documents]
        }

        for attribute in self._attributes:
            result[attribute.name] = []
            for document in self._documents:
                if kind is None:
                    nuggets: Optional[List[ASETNugget]] = None
                    if attribute.name in document.attribute_mappings.keys():
                        nuggets: List[ASETNugget] = document.attribute_mappings[attribute.name]
                elif kind == "text":
                    nuggets: Optional[List[str]] = None
                    if attribute.name in document.attribute_mappings.keys():
                        nuggets: List[str] = [n.text for n in document.attribute_mappings[attribute.name]]
                elif kind == "value":
                    nuggets: Optional[List[Optional[str]]] = None
                    if attribute.name in document.attribute_mappings.keys():
                        nuggets: List[Optional[str]] = [n.value for n in document.attribute_mappings[attribute.name]]
                else:
                    assert False, f"Unknown parameter kind '{kind}'!"
                result[attribute.name].append(nuggets)

        return result

    def validate_consistency(self) -> None:
        """Validate the consistency of the document base."""
        tick: float = time.time()
        # check that the document names are unique
        assert len([d.name for d in self._documents]) == len(set([d.name for d in self._documents]))

        # check that the attribute names are unique
        assert len([a.name for a in self._attributes]) == len(set([a.name for a in self._attributes]))

        # check that all nuggets in a document refer to that document
        for document in self._documents:
            for nugget in document.nuggets:
                assert nugget.document is document

        # check that the nuggets' span indices are valid
        for nugget in self.nuggets:
            assert 0 <= nugget.start_char < nugget.end_char <= len(nugget.document.text)

        # check that all attribute names in attribute mappings are part of the document base
        for document in self._documents:
            for attribute_name in document.attribute_mappings.keys():
                assert attribute_name in [attribute.name for attribute in self._attributes]

        # check that all nuggets referred to in attribute mappings are part of the document
        for document in self._documents:
            for nuggets in document.attribute_mappings.values():
                for nugget in nuggets:
                    for nug in document.nuggets:
                        if nug is nugget:
                            break
                    else:
                        assert False

        # check that all nugget signals are stored under their own signal string
        for nugget in self.nuggets:
            for signal_str, signal in nugget.signals.items():
                assert signal_str == signal.signal_str

        # check that all attribute signals are stored under their own signal string
        for attribute in self._attributes:
            for signal_str, signal in attribute.signals.items():
                assert signal_str == signal.signal_str

        # check that all document annotations are stored under their own annotation string
        for document in self._documents:
            for annotation_str, annotation in document.annotations.items():
                assert annotation_str == annotation.annotation_str

        tack: float = time.time()
        logger.info(f"Validated document base consistency in {tack - tick} seconds.")

    def to_bson(self) -> bytes:
        """
        Serialize the document base to a BSON byte string.

        https://pymongo.readthedocs.io/en/stable/api/bson/index.html

        :return: BSON byte representation of the document base
        """
        tick: float = time.time()

        # serialize the document base
        serializable_base: Dict[str, Any] = {
            "documents": [],
            "attributes": []
        }

        for attribute in self._attributes:
            # serialize the attribute
            serializable_attribute: Dict[str, Any] = {
                "name": attribute.name,
                "signals": {}
            }

            # serialize the signals
            for signal_str, signal in attribute.signals.items():
                if signal.do_serialize:
                    serializable_attribute["signals"][signal_str] = signal.to_serializable()

            serializable_base["attributes"].append(serializable_attribute)

        for document in self._documents:
            # serialize the document
            serializable_document: Dict[str, Any] = {
                "name": document.name,
                "text": document.text,
                "nuggets": [],
                "attribute_mappings": {},
                "annotations": {}
            }

            # serialize the attribute mappings
            for name, nuggets in document.attribute_mappings.items():
                nugget_ids: List[int] = []
                for nugget in nuggets:
                    for idx, doc_nugget in enumerate(document.nuggets):
                        if nugget is doc_nugget:
                            nugget_ids.append(idx)
                            break
                    else:
                        assert False, "The document does not contain the nugget that is assigned to the attribute."

                serializable_document["attribute_mappings"][name] = nugget_ids

            # serialize the annotations
            for annotation_str, annotation in document.annotations.items():
                if annotation.do_serialize:
                    serializable_document["annotations"][annotation_str] = annotation.to_serializable()

            for nugget in document.nuggets:
                # serialize the nugget
                serializable_nugget: Dict[str, Any] = {
                    "start_char": nugget.start_char,
                    "end_char": nugget.end_char,
                    "extractor_str": nugget.extractor_str,
                    "type_str": nugget.type_str,
                    "value": nugget.value,
                    "signals": {}
                }

                # serialize the signals
                for signal_str, signal in nugget.signals.items():
                    if signal.do_serialize:
                        serializable_nugget["signals"][signal_str] = signal.to_serializable()

                serializable_document["nuggets"].append(serializable_nugget)
            serializable_base["documents"].append(serializable_document)

        bson_bytes: bytes = bson.encode(serializable_base)

        tack: float = time.time()
        logger.info(f"Serialized document base in {tack - tick} seconds.")

        return bson_bytes

    @classmethod
    def from_bson(cls, bson_bytes: bytes) -> "ASETDocumentBase":
        """
        Deserialize a document base from a BSON byte string.

        https://pymongo.readthedocs.io/en/stable/api/bson/index.html

        :return: document base created from the BSON byte string
        """
        tick: float = time.time()

        serialized_base: Dict[str, Any] = bson.decode(bson_bytes)

        # deserialize the document base
        document_base: "ASETDocumentBase" = cls([], [])

        for serialized_attribute in serialized_base["attributes"]:
            # deserialize the attribute
            attribute: ASETAttribute = ASETAttribute(
                name=serialized_attribute["name"]
            )

            # deserialize the signals
            for signal_str, serialized_signal in serialized_attribute["signals"].items():
                attribute.signals[signal_str] = BaseSignal.from_serializable(serialized_signal, signal_str)

            document_base.attributes.append(attribute)

        for serialized_document in serialized_base["documents"]:
            # deserialize the document
            document: ASETDocument = ASETDocument(
                name=serialized_document["name"],
                text=serialized_document["text"]
            )

            for serialized_nugget in serialized_document["nuggets"]:
                # deserialize the nugget
                nugget: ASETNugget = ASETNugget(
                    document=document,
                    start_char=serialized_nugget["start_char"],
                    end_char=serialized_nugget["end_char"],
                    extractor_str=serialized_nugget["extractor_str"],
                    type_str=serialized_nugget["type_str"],
                    value=serialized_nugget["value"]
                )

                # deserialize the signals
                for signal_str, serialized_signal in serialized_nugget["signals"].items():
                    nugget.signals[signal_str] = BaseSignal.from_serializable(serialized_signal, signal_str)

                document.nuggets.append(nugget)

            # deserialize the attribute mappings
            for name, indices in serialized_document["attribute_mappings"].items():
                document.attribute_mappings[name] = [document.nuggets[idx] for idx in indices]

            # deserialize the annotations
            for annotation_str, serialized_annotation in serialized_document["annotations"].items():
                annotation: BaseAnnotation = BaseAnnotation.from_serializable(serialized_annotation, annotation_str)
                document.annotations[annotation_str] = annotation

            document_base.documents.append(document)

        tack: float = time.time()
        logger.info(f"Deserialized document base in {tack - tick} seconds.")

        return document_base
