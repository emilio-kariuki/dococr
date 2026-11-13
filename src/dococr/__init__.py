"""dococr: tell it the document type and the fields you want; get them back as JSON."""

from dococr.doctypes import DocTypeError, list_doc_types, load_doc_type
from dococr.extractors.base import Extractor, ExtractorError
from dococr.imaging import PageLimitError, SourceError
from dococr.models import (
    DocType,
    ExtractionResult,
    ExtractorOutput,
    FieldResult,
    FieldSpec,
    FieldType,
    ValuePart,
)
from dococr.pipeline import extract, get_extractor, warm_up

__all__ = [
    "DocType",
    "DocTypeError",
    "ExtractionResult",
    "Extractor",
    "ExtractorError",
    "ExtractorOutput",
    "FieldResult",
    "FieldSpec",
    "FieldType",
    "PageLimitError",
    "SourceError",
    "ValuePart",
    "extract",
    "get_extractor",
    "list_doc_types",
    "load_doc_type",
    "warm_up",
]
__version__ = "0.1.0"
