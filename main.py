import json
import logging
import sys

import datasets.aviation.aviation as dataset
from aset.data.annotations import SentenceStartCharsAnnotation
from aset.data.data import ASETAttribute, ASETDocument, ASETDocumentBase
from aset.matching.distance import SignalsMeanDistance
from aset.matching.phase import TreeSearchMatchingPhase
from aset.preprocessing.embedding import BERTContextSentenceEmbedder, FastTextLabelEmbedder, RelativePositionEmbedder, \
    SBERTExamplesEmbedder, SBERTTextEmbedder
from aset.preprocessing.extraction import StanzaNERExtractor
from aset.preprocessing.phase import PreprocessingPhase
from aset.resources import ResourceManager
from aset.statistics import Statistics
from aset.status import StatusFunction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":
    with ResourceManager() as resource_manager:
        status_function = StatusFunction(callback_fn=None)
        statistics = Statistics(do_collect=True)

        documents = dataset.load_dataset()

        # create the document base
        document_base = ASETDocumentBase(
            documents=[ASETDocument(document["id"], document["text"]) for document in documents],
            attributes=[
                ASETAttribute("date"),
                ASETAttribute("airport code")
            ]
        )

        # preprocessing
        preprocessing_phase = PreprocessingPhase(
            extractors=[
                StanzaNERExtractor()
            ],
            normalizers=[],
            embedders=[
                FastTextLabelEmbedder("FastTextEmbedding100000", True, [" ", "_"]),
                SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
                BERTContextSentenceEmbedder("BertLargeCasedResource"),
                RelativePositionEmbedder()
            ]
        )
        preprocessing_phase(document_base, status_function, statistics["preprocessing"])

        # matching
        matching_phase = TreeSearchMatchingPhase(
            distance=SignalsMeanDistance(
                signal_strings=[
                    "LabelEmbeddingSignal",
                    "TextEmbeddingSignal",
                    "ContextSentenceEmbeddingSignal",
                    "RelativePositionSignal",
                    "POSTagsSignal"
                ]
            ),
            examples_embedder=SBERTExamplesEmbedder("SBERTBertLargeNliMeanTokensResource"),
            max_num_feedback=10,
            max_children=2,
            max_distance=0.6,
            exploration_factor=1.2
        )


        def feedback_fn(feedback_request):
            if feedback_request["message"] == "give-feedback":
                nugget = feedback_request["nugget"]
                attribute = feedback_request["attribute"]

                # determine the context sentence
                sent_start_chars = nugget.document.annotations[SentenceStartCharsAnnotation.annotation_str].value
                context_start_char = 0
                context_end_char = 0
                for ix, sent_start_char in enumerate(sent_start_chars):
                    if sent_start_char > nugget.start_char:
                        if ix == 0:
                            context_start_char = 0
                            context_end_char = sent_start_char
                            break
                        else:
                            context_start_char = sent_start_chars[ix - 1]
                            context_end_char = sent_start_char
                            break
                else:
                    if sent_start_chars != []:
                        context_start_char = sent_start_chars[-1]
                        context_end_char = len(nugget.document.text)
                context_sentence = nugget.document.text[context_start_char:context_end_char]

                # get user feedback
                sys.stdout.flush()
                sys.stderr.flush()
                print("{}? '{}' from '{}'".format(attribute.name, nugget.text.ljust(40), context_sentence))

                while True:
                    s: str = input("y/n? ")
                    if s == "y":
                        return {"feedback": True}
                    elif s == "n":
                        return {"feedback": False}

            elif feedback_request["message"] == "give-examples":
                attribute = feedback_request["attribute"]

                sys.stdout.flush()
                sys.stderr.flush()
                print("Provide examples for '{}', leave empty to continue:".format(attribute.name))

                examples = []
                while True:
                    s = input("- ")
                    if s == "":
                        return {"examples": examples}
                    else:
                        examples.append(s)
            else:
                assert False, f"Unknown message '{feedback_request['message']}'!"


        matching_phase(document_base, feedback_fn, status_function, statistics["matching"])

        # store the result
        with open("cache/matched-document-base.bson", "wb") as file:
            file.write(document_base.to_bson())

    with open("cache/statistics.json", "w") as file:
        json.dump(statistics.to_serializable(), file, indent=4)
