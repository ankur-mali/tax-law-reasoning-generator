"""
Utility modules for Tax Law Reasoning Generator
Provides data structures, validation, and export functionality
"""

from .data_structures import (
    TaxEntity,
    TaxEvent,
    ReasoningStepData,
    TaxLawCase,
    EvaluationMetrics,
    DatasetSummary,
    ComplexityLevel,
    ReasoningStep,
    EntityType,
    EventCategory,
    serialize_case,
    deserialize_case,
    merge_cases,
)

from .validators import (
    validate_entity,
    validate_event,
    validate_reasoning_chain,
    validate_tax_case,
    ValidationError,
    EntityValidator,
    EventValidator,
    CaseValidator,
)

from .exporters import (
    export_to_json,
    export_to_csv,
    export_to_huggingface,
    export_dataset,
    DatasetExporter,
    export_to_jsonlines,
    export_evaluation_metrics,
)

__all__ = [
    # Data structures
    "TaxEntity",
    "TaxEvent",
    "ReasoningStepData",
    "TaxLawCase",
    "EvaluationMetrics",
    "DatasetSummary",
    "ComplexityLevel",
    "ReasoningStep",
    "EntityType",
    "EventCategory",

    # Data structure utilities
    "serialize_case",
    "deserialize_case",
    "merge_cases",

    # Validation
    "validate_entity",
    "validate_event",
    "validate_reasoning_chain",
    "validate_tax_case",
    "ValidationError",
    "EntityValidator",
    "EventValidator",
    "CaseValidator",

    # Export functionality
    "export_to_json",
    "export_to_csv",
    "export_to_huggingface",
    "export_dataset",
    "DatasetExporter",
    "export_to_jsonlines",
    "export_evaluation_metrics",
]
