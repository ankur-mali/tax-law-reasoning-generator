"""
Core Data Structures for Tax Law Reasoning Generator
Provides centralized data models for consistency across the system
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import json
import uuid


class ComplexityLevel(Enum):
    """Complexity levels for generated cases"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ReasoningStep(Enum):
    """Types of reasoning steps in tax law cases"""
    FACT_IDENTIFICATION = "fact_identification"
    RULE_APPLICATION = "rule_application"
    CALCULATION = "calculation"
    INTERPRETATION = "interpretation"
    CONCLUSION = "conclusion"


class EntityType(Enum):
    """Types of tax entities"""
    INDIVIDUAL = "individual"
    CORPORATION = "corporation"
    PARTNERSHIP = "partnership"
    TRUST = "trust"
    ESTATE = "estate"
    LLC = "llc"


class EventCategory(Enum):
    """Categories of tax events"""
    INCOME = "income"
    DEDUCTION = "deduction"
    CREDIT = "credit"
    PENALTY = "penalty"
    ADJUSTMENT = "adjustment"


@dataclass
class TaxEntity:
    """Represents a tax entity (individual, corporation, etc.)"""
    id: str
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate entity data after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.name:
            self.name = f"{self.entity_type.capitalize()}_{self.id[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaxEntity':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TaxEvent:
    """Represents a tax-relevant event or transaction"""
    id: str
    event_type: str
    amount: Optional[float] = None
    date: Optional[str] = None
    description: str = ""
    entities_involved: List[str] = field(default_factory=list)
    tax_implications: Dict[str, Any] = field(default_factory=dict)
    source_documents: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate event data after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d")
        if not self.description:
            self.description = f"Tax event: {self.event_type}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaxEvent':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ReasoningStepData:
    """Represents a single step in the reasoning chain"""
    step_id: str
    step_type: ReasoningStep
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning_text: str
    confidence_score: Optional[float] = None
    citations: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate reasoning step data after initialization"""
        if not self.step_id:
            self.step_id = str(uuid.uuid4())
        if self.confidence_score is not None:
            self.confidence_score = max(0.0, min(1.0, self.confidence_score))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['step_type'] = self.step_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStepData':
        """Create from dictionary"""
        if 'step_type' in data and isinstance(data['step_type'], str):
            data['step_type'] = ReasoningStep(data['step_type'])
        return cls(**data)


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating generated cases"""
    case_id: str
    complexity_level: str

    # Structure metrics
    num_entities: int
    num_events: int
    num_reasoning_steps: int
    narrative_length: int

    # Quality metrics
    narrative_coherence_score: float = 0.0
    reasoning_validity_score: float = 0.0
    tax_law_accuracy_score: float = 0.0
    overall_quality_score: float = 0.0

    # Difficulty metrics
    estimated_difficulty: float = 0.0
    human_solvability_score: float = 0.0

    # AI-specific metrics
    ai_enhancement_score: float = 0.0
    ai_validation_score: float = 0.0

    # Metadata
    generation_time: float = 0.0
    evaluation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TaxLawCase:
    """Complete tax law case with narrative and reasoning chain"""
    case_id: str
    title: str
    narrative: str
    entities: List[TaxEntity]
    events: List[TaxEvent]
    complexity_level: ComplexityLevel
    reasoning_chain: List[ReasoningStepData]
    ground_truth_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional AI enhancement data
    ai_enhanced: bool = False
    ai_validation_results: Optional[Dict[str, Any]] = None
    ai_complexity_assessment: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate case data after initialization"""
        if not self.case_id:
            self.case_id = str(uuid.uuid4())
        if not self.title:
            self.title = f"Tax Case {self.case_id[:8]}"
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['complexity_level'] = self.complexity_level.value
        data['entities'] = [entity.to_dict() for entity in self.entities]
        data['events'] = [event.to_dict() for event in self.events]
        data['reasoning_chain'] = [step.to_dict() for step in self.reasoning_chain]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaxLawCase':
        """Create from dictionary"""
        # Convert complexity level
        if 'complexity_level' in data and isinstance(data['complexity_level'], str):
            data['complexity_level'] = ComplexityLevel(data['complexity_level'])

        # Convert entities
        if 'entities' in data:
            data['entities'] = [TaxEntity.from_dict(e) for e in data['entities']]

        # Convert events
        if 'events' in data:
            data['events'] = [TaxEvent.from_dict(e) for e in data['events']]

        # Convert reasoning chain
        if 'reasoning_chain' in data:
            data['reasoning_chain'] = [ReasoningStepData.from_dict(s) for s in data['reasoning_chain']]

        return cls(**data)

    def get_entity_by_id(self, entity_id: str) -> Optional[TaxEntity]:
        """Get entity by ID"""
        return next((e for e in self.entities if e.id == entity_id), None)

    def get_events_by_entity(self, entity_id: str) -> List[TaxEvent]:
        """Get all events involving a specific entity"""
        return [e for e in self.events if entity_id in e.entities_involved]

    def calculate_total_income(self) -> float:
        """Calculate total income from all income events"""
        return sum(e.amount for e in self.events
                   if "income" in e.event_type.lower() and e.amount is not None)

    def calculate_total_deductions(self) -> float:
        """Calculate total deductions from all deduction events"""
        return sum(e.amount for e in self.events
                   if "deduction" in e.event_type.lower() and e.amount is not None)

    def get_complexity_score(self) -> float:
        """Calculate complexity score based on case elements"""
        base_score = {
            ComplexityLevel.BASIC: 0.25,
            ComplexityLevel.INTERMEDIATE: 0.50,
            ComplexityLevel.ADVANCED: 0.75,
            ComplexityLevel.EXPERT: 1.00
        }.get(self.complexity_level, 0.50)

        # Adjust based on actual complexity
        entity_factor = min(len(self.entities) / 4.0, 1.0) * 0.2
        event_factor = min(len(self.events) / 8.0, 1.0) * 0.2
        reasoning_factor = min(len(self.reasoning_chain) / 10.0, 1.0) * 0.1

        return min(base_score + entity_factor + event_factor + reasoning_factor, 1.0)


@dataclass
class DatasetSummary:
    """Summary information for a generated dataset"""
    dataset_id: str
    name: str
    description: str
    total_cases: int
    complexity_distribution: Dict[str, int]
    generation_config: Dict[str, Any]
    quality_metrics: Dict[str, float]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSummary':
        """Create from dictionary"""
        return cls(**data)


# Utility functions for working with data structures
def serialize_case(case: TaxLawCase) -> str:
    """Serialize case to JSON string"""
    return json.dumps(case.to_dict(), indent=2, ensure_ascii=False)


def deserialize_case(json_str: str) -> TaxLawCase:
    """Deserialize case from JSON string"""
    data = json.loads(json_str)
    return TaxLawCase.from_dict(data)


def merge_cases(*cases: TaxLawCase) -> TaxLawCase:
    """Merge multiple cases into one complex case"""
    if not cases:
        raise ValueError("At least one case required for merging")

    base_case = cases[0]
    merged_case = TaxLawCase(
        case_id=str(uuid.uuid4()),
        title=f"Merged Case: {', '.join(c.title for c in cases)}",
        narrative=" ".join(c.narrative for c in cases),
        entities=[],
        events=[],
        complexity_level=ComplexityLevel.EXPERT,  # Merged cases are always expert level
        reasoning_chain=[],
        ground_truth_answer="",
        metadata={
            'merged_from': [c.case_id for c in cases],
            'merge_timestamp': datetime.now().isoformat()
        }
    )

    # Combine entities, events, and reasoning steps
    for case in cases:
        merged_case.entities.extend(case.entities)
        merged_case.events.extend(case.events)
        merged_case.reasoning_chain.extend(case.reasoning_chain)

    return merged_case
