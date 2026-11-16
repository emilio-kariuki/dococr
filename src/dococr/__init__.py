"""dococr: tell it the document type (or let it detect it) and the fields you want; get them back as JSON."""

from dococr.detection import UnknownDocumentError
from dococr.doctypes import DocTypeError, list_doc_types, load_doc_type
from dococr.extractors.base import Extractor, ExtractorError
from dococr.imaging import PageLimitError, SourceError
from dococr.models import (
    Detection,
    DetectionCandidate,
    DocType,
    ExtractionResult,
    Evidence,
    ExtractorOutput,
    FieldResult,
    FieldSpec,
    FieldType,
    ValuePart,
)
from dococr.pipeline import detect, extract, get_extractor, warm_up

__all__ = [
    "Detection",
    "DetectionCandidate",
    "DocType",
    "DocTypeError",
    "ExtractionResult",
    "Extractor",
    "ExtractorError",
    "Evidence",
    "ExtractorOutput",
    "FieldResult",
    "FieldSpec",
    "FieldType",
    "PageLimitError",
    "SourceError",
    "UnknownDocumentError",
    "ValuePart",
    "detect",
    "extract",
    "get_extractor",
    "list_doc_types",
    "load_doc_type",
    "warm_up",
]
__version__ = "0.1.0"
