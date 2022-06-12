from typing import List

import pytest

from aset.data.data import ASETAttribute, ASETDocument, ASETDocumentBase, ASETNugget
from aset.data.signals import CachedDistanceSignal, LabelSignal, SentenceStartCharsSignal, CurrentMatchIndexSignal


@pytest.fixture
def aset_documents() -> List[ASETDocument]:
    return [
        ASETDocument(
            "document-0",
            "Wilhelm Conrad Röntgen (/ˈrɛntɡən, -dʒən, ˈrʌnt-/; [ˈvɪlhɛlm ˈʁœntɡən]; 27 March 1845 – 10 "
            "February 1923) was a German physicist, who, on 8 November 1895, produced and detected "
            "electromagnetic radiation in a wavelength range known as X-rays or Röntgen rays, an achievement "
            "that earned him the first Nobel Prize in Physics in 1901. In honour of his accomplishments, in "
            "2004 the International Union of Pure and Applied Chemistry (IUPAC) named element 111, "
            "roentgenium, a radioactive element with multiple unstable isotopes, after him."
        ),
        ASETDocument(
            "document-1",
            "Wilhelm Carl Werner Otto Fritz Franz Wien ([ˈviːn]; 13 January 1864 – 30 August 1928) was a "
            "German physicist who, in 1893, used theories about heat and electromagnetism to deduce Wien's "
            "displacement law, which calculates the emission of a blackbody at any temperature from the "
            "emission at any one reference temperature. He also formulated an expression for the black-body "
            "radiation which is correct in the photon-gas limit. His arguments were based on the notion of "
            "adiabatic invariance, and were instrumental for the formulation of quantum mechanics. Wien "
            "received the 1911 Nobel Prize for his work on heat radiation. He was a cousin of Max Wien, "
            "inventor of the Wien bridge."
        ),
        ASETDocument(
            "document-2",
            "Heike Kamerlingh Onnes ([ˈɔnəs]; 21 September 1853 – 21 February 1926) was a Dutch physicist and "
            "Nobel laureate. He exploited the Hampson-Linde cycle to investigate how materials behave when "
            "cooled to nearly absolute zero and later to liquefy helium for the first time. His production "
            "of extreme cryogenic temperatures led to his discovery of superconductivity in 1911: for "
            "certain materials, electrical resistance abruptly vanishes at very low temperatures."
        ),
        ASETDocument(
            "document-2",
            "Heike Kamerlingh Onnes ([ˈɔnəs]; 21 September 1853 – 21 February 1926) was a Dutch physicist and "
            "Nobel laureate. He exploited the Hampson-Linde cycle to investigate how materials behave when "
            "cooled to nearly absolute zero and later to liquefy helium for the first time. His production "
            "of extreme cryogenic temperatures led to his discovery of superconductivity in 1911: for "
            "certain materials, electrical resistance abruptly vanishes at very low temperatures."
        )
    ]


@pytest.fixture
def aset_nuggets(aset_documents) -> List[ASETNugget]:
    return [
        ASETNugget(aset_documents[0], 0, 22),
        ASETNugget(aset_documents[0], 56, 123),
        ASETNugget(aset_documents[1], 165, 176),
        ASETNugget(aset_documents[1], 234, 246),
        ASETNugget(aset_documents[2], 434, 456),
        ASETNugget(aset_documents[2], 123, 234),
        ASETNugget(aset_documents[2], 123, 234)
    ]


@pytest.fixture
def aset_attributes() -> List[ASETAttribute]:
    return [
        ASETAttribute("name"),
        ASETAttribute("field"),
        ASETAttribute("field")
    ]


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

    # test signals
    aset_nuggets[5][LabelSignal] = "my-label-signal"
    assert aset_nuggets[5].signals[LabelSignal.identifier].value == "my-label-signal"
    assert aset_nuggets[5][LabelSignal.identifier] == "my-label-signal"
    assert aset_nuggets[5][LabelSignal] == "my-label-signal"
    assert aset_nuggets[5] != aset_nuggets[6]
    assert aset_nuggets[6] != aset_nuggets[5]

    aset_nuggets[5][LabelSignal] = "new-value"
    assert aset_nuggets[5][LabelSignal] == "new-value"

    aset_nuggets[5][LabelSignal.identifier] = "new-new-value"
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
    aset_attributes[1][LabelSignal] = "my-label-signal"
    assert aset_attributes[1].signals[LabelSignal.identifier].value == "my-label-signal"
    assert aset_attributes[1][LabelSignal.identifier] == "my-label-signal"
    assert aset_attributes[1][LabelSignal] == "my-label-signal"
    assert aset_attributes[1] != aset_attributes[2]
    assert aset_attributes[2] != aset_attributes[1]

    aset_attributes[1][LabelSignal] = "new-value"
    assert aset_attributes[1][LabelSignal] == "new-value"

    aset_attributes[1][LabelSignal.identifier] = "new-new-value"
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

    # test signals
    aset_documents[2][SentenceStartCharsSignal] = [0, 10, 20]
    assert aset_documents[2].signals[SentenceStartCharsSignal.identifier].value == [0, 10, 20]
    assert aset_documents[2][SentenceStartCharsSignal.identifier] == [0, 10, 20]
    assert aset_documents[2][SentenceStartCharsSignal] == [0, 10, 20]
    assert aset_documents[2] != aset_documents[3]
    assert aset_documents[3] != aset_documents[2]

    aset_documents[2][SentenceStartCharsSignal] = [1, 2, 3]
    assert aset_documents[2][SentenceStartCharsSignal] == [1, 2, 3]

    aset_documents[2][SentenceStartCharsSignal.identifier] = [3, 4, 5]
    assert aset_documents[2][SentenceStartCharsSignal] == [3, 4, 5]

    aset_documents[2][SentenceStartCharsSignal.identifier] = SentenceStartCharsSignal([6, 7])
    assert aset_documents[2][SentenceStartCharsSignal] == [6, 7]

    aset_documents[2][CurrentMatchIndexSignal] = CurrentMatchIndexSignal(2)
    assert aset_documents[2][CurrentMatchIndexSignal] == 2


def test_aset_document_base(aset_documents, aset_nuggets, aset_attributes, aset_document_base) -> None:
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

    # test get_nuggets_for_attribute
    assert aset_document_base.get_nuggets_for_attribute(aset_attributes[0]) == [aset_nuggets[0]]

    # test get_column_for_attribute
    assert aset_document_base.get_column_for_attribute(aset_attributes[0]) == [[aset_nuggets[0]], None, None]

    # test validate_consistency
    assert aset_document_base.validate_consistency()

    # test to_bson and from_bson
    bson_bytes: bytes = aset_document_base.to_bson()
    copied_aset_document_base: ASETDocumentBase = ASETDocumentBase.from_bson(bson_bytes)
    assert aset_document_base == copied_aset_document_base
    assert copied_aset_document_base == aset_document_base
