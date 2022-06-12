import csv
import glob
import json
import logging
from json import JSONDecodeError

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from bson import InvalidBSON

from aset.configuration import ASETPipeline
from aset.data.data import ASETAttribute, ASETDocument, ASETDocumentBase
from aset.interaction import EmptyInteractionCallback, InteractionCallback
from aset.matching.distance import SignalsMeanDistance
from aset.matching.matching import RankingBasedMatcher
from aset.preprocessing.embedding import BERTContextSentenceEmbedder, FastTextLabelEmbedder, RelativePositionEmbedder, \
    SBERTTextEmbedder
from aset.preprocessing.extraction import StanzaNERExtractor
from aset.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, \
    CopyAttributeNameLabelParaphraser, SplitAttributeNameLabelParaphraser
from aset.preprocessing.normalization import VerySimpleDateNormalizer
from aset.preprocessing.other_processing import ContextSentenceCacher
from aset.statistics import Statistics
from aset.status import StatusCallback

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class ASETAPI(QObject):
    def __init__(self, feedback_mutex, feedback_cond):
        super(ASETAPI, self).__init__()
        self.feedback = None
        self.feedback_mutex = feedback_mutex
        self.feedback_cond = feedback_cond
        logger.info("Initialized ASETAPI.")

    #########################################
    # signals (aset api --> aset ui)
    #########################################
    status = pyqtSignal(str, float)  # message, progress
    finished = pyqtSignal(str)  # message
    error = pyqtSignal(str)  # message
    document_base_to_ui = pyqtSignal(ASETDocumentBase)  # document base
    preprocessing_phase_to_ui = pyqtSignal(ASETPipeline)  # preprocessing phase
    matching_phase_to_ui = pyqtSignal(ASETPipeline)  # matching phase
    statistics_to_ui = pyqtSignal(Statistics)  # statistics
    feedback_request_to_ui = pyqtSignal(dict)

    ##############################
    # slots (aset ui --> aset api)
    ##############################
    @pyqtSlot(str, list)
    def create_document_base(self, path, attribute_names):
        logger.debug("Called slot 'create_document_base'.")
        self.status.emit("Creating document base...", -1)
        try:
            if path == "":
                logger.error("The path cannot be empty!")
                self.error.emit("The path cannot be empty!")
                return

            file_paths = glob.glob(path)
            documents = []
            for file_path in file_paths:
                with open(file_path, encoding="utf-8") as file:
                    documents.append(ASETDocument(file_path, file.read()))

            if len(set(attribute_names)) != len(attribute_names):
                logger.error("Attribute names must be unique!")
                self.error.emit("Attribute names must be unique!")
                return

            for attribute_name in attribute_names:
                if attribute_name == "":
                    logger.error("Attribute names cannot be empty!")
                    self.error.emit("Attribute names cannot be empty!")
                    return

            attributes = []
            for attribute_name in attribute_names:
                attributes.append(ASETAttribute(attribute_name))

            document_base = ASETDocumentBase(documents, attributes)

            if not document_base.validate_consistency():
                logger.error("Document base is inconsistent!")
                self.error.emit("Document base is inconsistent!")
                return

            self.document_base_to_ui.emit(document_base)
            self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("Directory does not exist!")
            self.error.emit("Directory does not exist!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str, ASETDocumentBase)
    def add_attribute(self, name, document_base):
        logger.debug("Called slot 'add_attribute'.")
        self.status.emit("Adding attribute...", -1)
        try:
            if name in [attribute.name for attribute in document_base.attributes]:
                logger.error("Attribute name already exists!")
                self.error.emit("Attribute name already exists!")
            elif name == "":
                logger.error("Attribute name must not be empty!")
                self.error.emit("Attribute name must not be empty!")
            else:
                document_base.attributes.append(ASETAttribute(name))
                self.document_base_to_ui.emit(document_base)
                self.finished.emit("Finished!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(list, ASETDocumentBase)
    def add_attributes(self, names, document_base):
        logger.debug("Called slot 'add_attributes'.")
        self.status.emit("Adding attributes...", -1)
        try:
            already_existing_names = []
            for name in names:
                if name in [attribute.name for attribute in document_base.attributes]:
                    logger.info(f"Attribute name '{name}' already exists and was thus not added.")
                    already_existing_names.append(name)
                elif name == "":
                    logger.info("Attribute name must not be empty and was thus ignored.")
                else:
                    document_base.attributes.append(ASETAttribute(name))
            self.document_base_to_ui.emit(document_base)
            if already_existing_names == []:
                self.finished.emit("Finished!")
            elif len(already_existing_names) == 1:
                self.finished.emit(f"Attribute '{already_existing_names[0]}' already existed and was thus not added!")
            else:
                list_str = ", ".join(f"'{name}'" for name in already_existing_names)
                self.finished.emit(f"Attributes {list_str} already existed and were thus not added!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str, ASETDocumentBase)
    def forget_matches_for_attribute(self, name, document_base):
        logger.debug("Called slot 'forget_matches_for_attribute'.")
        self.status.emit("Forgetting matches...", -1)
        try:
            if name in [attribute.name for attribute in document_base.attributes]:
                for document in document_base.documents:
                    if name in document.attribute_mappings.keys():
                        del document.attribute_mappings[name]
                self.document_base_to_ui.emit(document_base)
                self.finished.emit("Finished!")
            else:
                logger.error("Attribute name does not exist!")
                self.error.emit("Attribute name does not exist!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str, ASETDocumentBase)
    def remove_attribute(self, name, document_base):
        logger.debug("Called slot 'remove_attribute'.")
        self.status.emit("Removing attribute...", -1)
        try:
            if name in [attribute.name for attribute in document_base.attributes]:
                for document in document_base.documents:
                    if name in document.attribute_mappings.keys():
                        del document.attribute_mappings[name]

                for attribute in document_base.attributes:
                    if attribute.name == name:
                        document_base.attributes.remove(attribute)
                        break
                self.document_base_to_ui.emit(document_base)
                self.finished.emit("Finished!")
            else:
                logger.error("Attribute name does not exist!")
                self.error.emit("Attribute name does not exist!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str)
    def load_document_base_from_bson(self, path):
        logger.debug("Called slot 'load_document_base_from_bson'.")
        self.status.emit("Loading document base from BSON...", -1)
        try:
            with open(path, "rb") as file:
                document_base = ASETDocumentBase.from_bson(file.read())

                if not document_base.validate_consistency():
                    logger.error("Document base is inconsistent!")
                    self.error.emit("Document base is inconsistent!")
                    return

                self.document_base_to_ui.emit(document_base)
                self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("File does not exist!")
            self.error.emit("File does not exist!")
        except InvalidBSON:
            logger.error("Unable to decode file!")
            self.error.emit("Unable to decode file!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str, ASETDocumentBase)
    def save_document_base_to_bson(self, path, document_base):
        logger.debug("Called slot 'save_document_base_to_bson'.")
        self.status.emit("Saving document base to BSON...", -1)
        try:
            with open(path, "wb") as file:
                file.write(document_base.to_bson())
                self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("Directory does not exist!")
            self.error.emit("Directory does not exist!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str, ASETDocumentBase)
    def save_table_to_csv(self, path, document_base):
        logger.debug("Called slot 'save_table_to_csv'.")
        self.status.emit("Saving table to CSV...", -1)
        try:
            # TODO: currently stores the text of the first matching nugget (if there is one)
            table_dict = document_base.to_table_dict("text")
            headers = list(table_dict.keys())
            rows = []
            for ix in range(len(table_dict[headers[0]])):
                row = []
                for header in headers:
                    if header == "document-name":
                        row.append(table_dict[header][ix])
                    elif table_dict[header][ix] == []:
                        row.append(None)
                    else:
                        row.append(table_dict[header][ix][0])
                rows.append(row)
            with open(path, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(headers)
                writer.writerows(rows)
            self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("Directory does not exist!")
            self.error.emit("Directory does not exist!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(ASETDocumentBase)
    def forget_matches(self, document_base):
        logger.debug("Called slot 'forget_matches'.")
        self.status.emit("Forgetting matches...", -1)
        try:
            for document in document_base.documents:
                document.attribute_mappings.clear()
            self.document_base_to_ui.emit(document_base)
            self.finished.emit("Finished!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str)
    def load_preprocessing_phase_from_config(self, path):
        logger.debug("Called slot 'load_preprocessing_phase_from_config'.")
        self.status.emit("Loading preprocessing phase from config...", -1)
        try:
            with open(path, "r", encoding="utf-8") as file:
                preprocessing_phase = ASETPipeline.from_config(json.load(file))
                self.preprocessing_phase_to_ui.emit(preprocessing_phase)
                self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("File does not exist!")
            self.error.emit("File does not exist!")
        except JSONDecodeError:
            logger.error("Unable to decode file!")
            self.error.emit("Unable to decode file!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str, ASETPipeline)
    def save_preprocessing_phase_to_config(self, path, preprocessing_phase):
        logger.debug("Called slot 'save_preprocessing_phase_to_config'.")
        self.status.emit("Saving preprocessing phase to config...", -1)
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(preprocessing_phase.to_config(), file, indent=2)
                self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("Directory does not exist!")
            self.error.emit("Directory does not exist!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str)
    def load_matching_phase_from_config(self, path):
        logger.debug("Called slot 'load_matching_phase_from_config'.")
        self.status.emit("Loading matching phase from config...", -1)
        try:
            with open(path, "r", encoding="utf-8") as file:
                matching_phase = ASETPipeline.from_config(json.load(file))
                self.matching_phase_to_ui.emit(matching_phase)
                self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("File does not exist!")
            self.error.emit("File does not exist!")
        except JSONDecodeError:
            logger.error("Unable to decode file!")
            self.error.emit("Unable to decode file!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str, ASETPipeline)
    def save_matching_phase_to_config(self, path, matching_phase):
        logger.debug("Called slot 'save_matching_phase_to_config'.")
        self.status.emit("Saving matching phase to config...", -1)
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(matching_phase.to_config(), file, indent=2)
                self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("Directory does not exist!")
            self.error.emit("Directory does not exist!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(ASETDocumentBase, ASETPipeline, Statistics)
    def run_preprocessing_phase(self, document_base, preprocessing_phase, statistics):
        logger.debug("Called slot 'run_preprocessing_phase'.")
        self.status.emit("Running preprocessing phase...", -1)
        try:
            def status_callback_fn(message, progress):
                self.status.emit(message, progress)

            status_callback = StatusCallback(status_callback_fn)

            preprocessing_phase(document_base, EmptyInteractionCallback(), status_callback, statistics)

            self.document_base_to_ui.emit(document_base)
            self.statistics_to_ui.emit(statistics)
            self.finished.emit("Finished!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(ASETDocumentBase, ASETPipeline, Statistics)
    def run_matching_phase(self, document_base, matching_phase, statistics):
        logger.debug("Called slot 'run_matching_phase'.")
        self.status.emit("Running matching phase...", -1)
        try:
            def status_callback_fn(message, progress):
                self.status.emit(message, progress)

            status_callback = StatusCallback(status_callback_fn)

            def interaction_callback_fn(pipeline_element_identifier, feedback_request):
                feedback_request["identifier"] = pipeline_element_identifier
                self.feedback_request_to_ui.emit(feedback_request)

                self.feedback_mutex.lock()
                try:
                    self.feedback_cond.wait(self.feedback_mutex)
                finally:
                    self.feedback_mutex.unlock()

                return self.feedback

            interaction_callback = InteractionCallback(interaction_callback_fn)

            matching_phase(document_base, interaction_callback, status_callback, statistics)

            self.document_base_to_ui.emit(document_base)
            self.statistics_to_ui.emit(statistics)
            self.finished.emit("Finished!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(str, Statistics)
    def save_statistics_to_json(self, path, statistics):
        logger.debug("Called slot 'save_statistics_to_json'.")
        self.status.emit("Saving statistics to JSON...", -1)
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(statistics.to_serializable(), file, indent=2)
                self.finished.emit("Finished!")
        except FileNotFoundError:
            logger.error("Directory does not exist!")
            self.error.emit("Directory does not exist!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(ASETDocumentBase, Statistics)
    def load_and_run_default_preprocessing_phase(self, document_base, statistics):
        logger.debug("Called slot 'load_and_run_default_preprocessing_phase'.")
        try:
            # load default preprocessing phase
            self.status.emit("Loading default preprocessing phase...", -1)

            preprocessing_phase = ASETPipeline(
                [
                    StanzaNERExtractor(),
                    VerySimpleDateNormalizer(),
                    OntoNotesLabelParaphraser(),
                    CopyAttributeNameLabelParaphraser(),
                    ContextSentenceCacher(),
                    FastTextLabelEmbedder("FastTextEmbedding100000", True, [" ", "_"]),
                    SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
                    BERTContextSentenceEmbedder("BertLargeCasedResource"),
                    RelativePositionEmbedder()
                ]
            )

            # run preprocessing phase
            def status_callback_fn(message, progress):
                self.status.emit(message, progress)

            status_callback = StatusCallback(status_callback_fn)

            preprocessing_phase(document_base, EmptyInteractionCallback(), status_callback, statistics)

            self.preprocessing_phase_to_ui.emit(preprocessing_phase)
            self.document_base_to_ui.emit(document_base)
            self.statistics_to_ui.emit(statistics)
            self.finished.emit("Finished!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))

    @pyqtSlot(ASETDocumentBase, Statistics)
    def load_and_run_default_matching_phase(self, document_base, statistics):
        logger.debug("Called slot 'load_and_run_default_matching_phase'.")
        try:
            # load default matching phase
            self.status.emit("Loading default matching phase...", -1)

            matching_phase = ASETPipeline(
                [
                    SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
                    ContextSentenceCacher(),
                    FastTextLabelEmbedder("FastTextEmbedding100000", True, [" ", "_"]),
                    RankingBasedMatcher(
                        distance=SignalsMeanDistance(
                            signal_identifiers=[
                                "LabelEmbeddingSignal",
                                "TextEmbeddingSignal",
                                "ContextSentenceEmbeddingSignal",
                                "RelativePositionSignal"
                            ]
                        ),
                        max_num_feedback=20,
                        len_ranked_list=10,
                        max_distance=0.2,
                        num_random_docs=1,
                        sampling_mode="AT_MAX_DISTANCE_THRESHOLD"
                    )
                ]
            )

            # run matching phase
            def status_callback_fn(message, progress):
                self.status.emit(message, progress)

            status_callback = StatusCallback(status_callback_fn)

            def interaction_callback_fn(pipeline_element_identifier, feedback_request):
                feedback_request["identifier"] = pipeline_element_identifier
                self.feedback_request_to_ui.emit(feedback_request)

                self.feedback_mutex.lock()
                try:
                    self.feedback_cond.wait(self.feedback_mutex)
                finally:
                    self.feedback_mutex.unlock()

                return self.feedback

            interaction_callback = InteractionCallback(interaction_callback_fn)

            matching_phase(document_base, interaction_callback, status_callback, statistics)

            self.matching_phase_to_ui.emit(matching_phase)
            self.document_base_to_ui.emit(document_base)
            self.statistics_to_ui.emit(statistics)
            self.finished.emit("Finished!")
        except Exception as e:
            logger.error(str(e))
            self.error.emit(str(e))
