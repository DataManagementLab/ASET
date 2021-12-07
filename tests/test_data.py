from typing import List

import pytest

from aset.data.annotations import CurrentMatchIndexAnnotation, SentenceStartCharsAnnotation
from aset.data.data import ASETAttribute, ASETDocument, ASETDocumentBase, ASETNugget
from aset.data.signals import CachedDistanceSignal, LabelSignal


@pytest.fixture
def aset_documents() -> List[ASETDocument]:
    document_0: ASETDocument = ASETDocument(
        "document-0",
        "Wilhelm Conrad Röntgen (/ˈrɛntɡən, -dʒən, ˈrʌnt-/; [ˈvɪlhɛlm ˈʁœntɡən]; 27 March 1845 – 10 "
        "February 1923) was a German physicist, who, on 8 November 1895, produced and detected "
        "electromagnetic radiation in a wavelength range known as X-rays or Röntgen rays, an achievement "
        "that earned him the first Nobel Prize in Physics in 1901. In honour of his accomplishments, in "
        "2004 the International Union of Pure and Applied Chemistry (IUPAC) named element 111, "
        "roentgenium, a radioactive element with multiple unstable isotopes, after him."
    )
    document_1: ASETDocument = ASETDocument(
        "document-1",
        "Wilhelm Carl Werner Otto Fritz Franz Wien ([ˈviːn]; 13 January 1864 – 30 August 1928) was a "
        "German physicist who, in 1893, used theories about heat and electromagnetism to deduce Wien's "
        "displacement law, which calculates the emission of a blackbody at any temperature from the "
        "emission at any one reference temperature. He also formulated an expression for the black-body "
        "radiation which is correct in the photon-gas limit. His arguments were based on the notion of "
        "adiabatic invariance, and were instrumental for the formulation of quantum mechanics. Wien "
        "received the 1911 Nobel Prize for his work on heat radiation. He was a cousin of Max Wien, "
        "inventor of the Wien bridge."
    )
    document_2: ASETDocument = ASETDocument(
        "document-2",
        "Heike Kamerlingh Onnes ([ˈɔnəs]; 21 September 1853 – 21 February 1926) was a Dutch physicist and "
        "Nobel laureate. He exploited the Hampson-Linde cycle to investigate how materials behave when "
        "cooled to nearly absolute zero and later to liquefy helium for the first time. His production "
        "of extreme cryogenic temperatures led to his discovery of superconductivity in 1911: for "
        "certain materials, electrical resistance abruptly vanishes at very low temperatures."
    )
    document_2.annotations[SentenceStartCharsAnnotation.annotation_str] = SentenceStartCharsAnnotation([0, 10, 20])
    document_3: ASETDocument = ASETDocument(
        "document-2",
        "Heike Kamerlingh Onnes ([ˈɔnəs]; 21 September 1853 – 21 February 1926) was a Dutch physicist and "
        "Nobel laureate. He exploited the Hampson-Linde cycle to investigate how materials behave when "
        "cooled to nearly absolute zero and later to liquefy helium for the first time. His production "
        "of extreme cryogenic temperatures led to his discovery of superconductivity in 1911: for "
        "certain materials, electrical resistance abruptly vanishes at very low temperatures."
    )
    return [document_0, document_1, document_2, document_3]


@pytest.fixture
def aset_nuggets(aset_documents) -> List[ASETNugget]:
    nugget_0: ASETNugget = ASETNugget(aset_documents[0], 0, 22, None, None, None)
    nugget_1: ASETNugget = ASETNugget(aset_documents[0], 56, 123, "demo-extractor", None, None)
    nugget_2: ASETNugget = ASETNugget(aset_documents[1], 165, 176, "demo-extractor", "demo-type", None)
    nugget_3: ASETNugget = ASETNugget(aset_documents[1], 234, 246, "demo-extractor", "demo-type", "demo-value")
    nugget_4: ASETNugget = ASETNugget(aset_documents[2], 434, 456, "demo-extractor", "demo-type", "demo-value")
    nugget_5: ASETNugget = ASETNugget(aset_documents[2], 123, 234, "demo-extractor", "demo-type", "demo-value")
    nugget_5.signals[LabelSignal.signal_str] = LabelSignal("my-label-signal")
    nugget_6: ASETNugget = ASETNugget(aset_documents[2], 123, 234, "demo-extractor", "demo-type", "demo-value")
    return [nugget_0, nugget_1, nugget_2, nugget_3, nugget_4, nugget_5, nugget_6]


@pytest.fixture
def aset_attributes() -> List[ASETAttribute]:
    attribute_0: ASETAttribute = ASETAttribute("name")
    attribute_1: ASETAttribute = ASETAttribute("field")
    attribute_1.signals[LabelSignal.signal_str] = LabelSignal("my-label-signal")
    attribute_2: ASETAttribute = ASETAttribute("field")
    return [attribute_0, attribute_1, attribute_2]


@pytest.fixture
def aset_document_base(aset_documents, aset_nuggets, aset_attributes) -> ASETDocumentBase:
    # link nuggets to documents
    for nugget in aset_nuggets:
        nugget.document.nuggets.append(nugget)

    # set up some dummy attribute mappings for documents 0 and 1
    aset_documents[0].attribute_mappings[aset_attributes[0].name] = [aset_nuggets[0]]
    aset_documents[0].attribute_mappings[aset_attributes[1].name] = []

    return ASETDocumentBase(
        documents=aset_documents[:-1],
        attributes=aset_attributes[:-1]
    )


def test_aset_nugget(aset_documents, aset_nuggets, aset_attributes, aset_document_base) -> None:
    # test __eq__
    assert aset_nuggets[0] == aset_nuggets[0]
    assert aset_nuggets[0] != aset_nuggets[1]
    assert aset_nuggets[1] != aset_nuggets[0]
    assert aset_nuggets[0] != object()
    assert object() != aset_nuggets[0]

    # test __str__ and __repr__ and __hash__
    for nugget in aset_nuggets:
        assert str(nugget) == f"'{nugget.text}'"
        assert repr(nugget) == f"ASETNugget({repr(nugget.document)}, {nugget.start_char}, {nugget.end_char})"
        assert hash(nugget) == hash((nugget.document, nugget.start_char, nugget.end_char))

    # test document
    assert aset_nuggets[0].document is aset_documents[0]

    # test start_char and end_char
    assert aset_nuggets[0].start_char == 0
    assert aset_nuggets[0].end_char == 22

    # test text
    assert aset_nuggets[0].text == "Wilhelm Conrad Röntgen"

    # test extractor_str
    assert aset_nuggets[1].extractor_str == "demo-extractor"

    # test type_str
    assert aset_nuggets[2].type_str == "demo-type"

    # test value
    assert aset_nuggets[3].value == "demo-value"

    # test signals
    assert aset_nuggets[5].signals[LabelSignal.signal_str].value == "my-label-signal"
    assert aset_nuggets[5] != aset_nuggets[6]
    assert aset_nuggets[6] != aset_nuggets[5]

    assert aset_nuggets[5][LabelSignal] == "my-label-signal"
    assert aset_nuggets[5][LabelSignal.signal_str] == "my-label-signal"

    aset_nuggets[5][LabelSignal] = "new-value"
    assert aset_nuggets[5][LabelSignal] == "new-value"

    aset_nuggets[5][LabelSignal.signal_str] = "new-new-value"
    assert aset_nuggets[5][LabelSignal] == "new-new-value"

    aset_nuggets[5][LabelSignal] = LabelSignal("another-value")
    assert aset_nuggets[5][LabelSignal] == "another-value"

    aset_nuggets[5][CachedDistanceSignal] = CachedDistanceSignal(0.23)
    assert aset_nuggets[5][CachedDistanceSignal] == 0.23


def test_aset_attribute(aset_documents, aset_nuggets, aset_attributes, aset_document_base) -> None:
    # test __eq__
    assert aset_attributes[0] == aset_attributes[0]
    assert aset_attributes[0] != aset_attributes[1]
    assert aset_attributes[1] != aset_attributes[0]
    assert aset_attributes[0] != object()
    assert object() != aset_attributes[0]

    # test __str__ and __repr__ and __hash__
    for attribute in aset_attributes:
        assert str(attribute) == f"'{attribute.name}'"
        assert repr(attribute) == f"ASETAttribute('{attribute.name}')"
        assert hash(attribute) == hash(attribute.name)

    # test name
    assert aset_attributes[0].name == "name"

    # test signals
    assert aset_attributes[1].signals[LabelSignal.signal_str].value == "my-label-signal"
    assert aset_attributes[1] != aset_attributes[2]
    assert aset_attributes[2] != aset_attributes[1]

    assert aset_attributes[1][LabelSignal] == "my-label-signal"
    assert aset_attributes[1][LabelSignal.signal_str] == "my-label-signal"

    aset_attributes[1][LabelSignal] = "new-value"
    assert aset_attributes[1][LabelSignal] == "new-value"

    aset_attributes[1][LabelSignal.signal_str] = "new-new-value"
    assert aset_attributes[1][LabelSignal] == "new-new-value"

    aset_attributes[1][LabelSignal] = LabelSignal("another-value")
    assert aset_attributes[1][LabelSignal] == "another-value"

    aset_attributes[1][CachedDistanceSignal] = CachedDistanceSignal(0.23)
    assert aset_attributes[1][CachedDistanceSignal] == 0.23


def test_aset_document(aset_documents, aset_nuggets, aset_attributes, aset_document_base) -> None:
    # test __eq__
    assert aset_documents[0] == aset_documents[0]
    assert aset_documents[0] != aset_documents[1]
    assert aset_documents[1] != aset_documents[0]
    assert aset_documents[0] != object()
    assert object() != aset_documents[0]

    # test __str__ and __repr__ and __hash__
    for document in aset_documents:
        assert str(document) == f"'{document.text}'"
        assert repr(document) == f"ASETDocument('{document.name}', '{document.text}')"
        assert hash(document) == hash(document.name)

    # test name
    assert aset_documents[0].name == "document-0"

    # test text
    assert aset_documents[0].text[:40] == "Wilhelm Conrad Röntgen (/ˈrɛntɡən, -dʒən"

    # test nuggets
    assert aset_documents[0].nuggets == [aset_nuggets[0], aset_nuggets[1]]

    # test attribute mappings
    assert aset_documents[0].attribute_mappings[aset_attributes[0].name] == [aset_nuggets[0]]
    assert aset_documents[0].attribute_mappings[aset_attributes[1].name] == []

    # test annotations
    assert aset_documents[2].annotations[SentenceStartCharsAnnotation.annotation_str].value == [0, 10, 20]
    assert aset_documents[2] != aset_documents[3]
    assert aset_documents[3] != aset_documents[2]

    assert aset_documents[2][SentenceStartCharsAnnotation] == [0, 10, 20]
    assert aset_documents[2][SentenceStartCharsAnnotation.annotation_str] == [0, 10, 20]

    aset_documents[2][SentenceStartCharsAnnotation] = [1, 2, 3]
    assert aset_documents[2][SentenceStartCharsAnnotation] == [1, 2, 3]

    aset_documents[2][SentenceStartCharsAnnotation.annotation_str] = [3, 4, 5]
    assert aset_documents[2][SentenceStartCharsAnnotation] == [3, 4, 5]

    aset_documents[2][SentenceStartCharsAnnotation] = SentenceStartCharsAnnotation([6, 7])
    assert aset_documents[2][SentenceStartCharsAnnotation] == [6, 7]

    aset_documents[2][CurrentMatchIndexAnnotation] = CurrentMatchIndexAnnotation(2)
    assert aset_documents[2][CurrentMatchIndexAnnotation] == 2


def test_document_base(aset_documents, aset_nuggets, aset_attributes, aset_document_base) -> None:
    # test __eq__
    assert aset_document_base == aset_document_base
    assert aset_document_base != ASETDocumentBase(aset_documents, aset_attributes[:1])
    assert ASETDocumentBase(aset_documents, aset_attributes[:1]) != aset_document_base
    assert aset_document_base != object()
    assert object() != aset_document_base

    # test __str__
    assert str(aset_document_base) == "(3 documents, 7 nuggets, 2 attributes)"

    # test __repr__
    assert repr(aset_document_base) == "ASETDocumentBase([{}], [{}])".format(
        ", ".join(repr(document) for document in aset_document_base.documents),
        ", ".join(repr(attribute) for attribute in aset_document_base.attributes)
    )

    # test documents
    assert aset_document_base.documents == aset_documents[:-1]

    # test attributes
    assert aset_document_base.attributes == aset_attributes[:-1]

    # test nuggets
    assert aset_document_base.nuggets == aset_nuggets

    # test to_table_dict
    assert aset_document_base.to_table_dict() == {
        "document-name": ["document-0", "document-1", "document-2"],
        "name": [[aset_nuggets[0]], None, None],
        "field": [[], None, None]
    }

    assert aset_document_base.to_table_dict("text") == {
        "document-name": ["document-0", "document-1", "document-2"],
        "name": [["Wilhelm Conrad Röntgen"], None, None],
        "field": [[], None, None]
    }

    assert aset_document_base.to_table_dict("value") == {
        "document-name": ["document-0", "document-1", "document-2"],
        "name": [[None], None, None],
        "field": [[], None, None]
    }

    # test validate_consistency
    aset_document_base.validate_consistency()

    # test to_bson and from_bson
    bson_bytes: bytes = aset_document_base.to_bson()
    copied_aset_document_base: ASETDocumentBase = ASETDocumentBase.from_bson(bson_bytes)
    assert aset_document_base == copied_aset_document_base
    assert copied_aset_document_base == aset_document_base
