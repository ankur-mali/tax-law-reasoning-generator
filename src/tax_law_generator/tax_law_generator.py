"""
Synthetic Tax Law Reasoning Data Generator
Based on MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning

This module generates synthetic tax law cases for evaluating GenAI reasoning capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
import uuid
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@dataclass
class TaxEntity:
    """Represents a tax entity (individual, corporation, etc.)"""
    id: str
    name: str
    entity_type: str  # individual, corporation, partnership, etc.
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaxEvent:
    """Represents a tax-relevant event or transaction"""
    id: str
    event_type: str  # income, deduction, credit, etc.
    amount: Optional[float] = None
    date: Optional[str] = None
    description: str = ""
    entities_involved: List[str] = field(default_factory=list)
    tax_implications: Dict[str, Any] = field(default_factory=dict)


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


class BaseGenerator(ABC):
    """Abstract base class for all generators"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """Generate synthetic data"""
        pass


class EntityGenerator(BaseGenerator):
    """Generates synthetic tax entities"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.entity_templates = {
            "individual": {
                "attributes": ["age", "income_level", "filing_status", "dependents"],
                "income_ranges": {"low": (0, 50000), "medium": (50001, 150000), "high": (150001, 500000)}
            },
            "corporation": {
                "attributes": ["industry", "revenue", "employees", "structure_type"],
                "revenue_ranges": {"small": (0, 1000000), "medium": (1000001, 50000000),
                                   "large": (50000001, 1000000000)}
            }
        }

    def generate(self, entity_type: str = "individual", **kwargs) -> TaxEntity:
        """Generate a tax entity"""
        entity_id = str(uuid.uuid4())
        name = f"{entity_type.capitalize()}_{entity_id[:8]}"

        # Generate realistic attributes based on entity type
        attributes = self._generate_attributes(entity_type)

        return TaxEntity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes
        )

    def _generate_attributes(self, entity_type: str) -> Dict[str, Any]:
        """Generate attributes specific to entity type"""
        import random

        template = self.entity_templates.get(entity_type, {})
        attributes = {}

        if entity_type == "individual":
            attributes.update({
                "age": random.randint(18, 80),
                "income_level": random.choice(["low", "medium", "high"]),
                "filing_status": random.choice(["single", "married_joint", "married_separate", "head_of_household"]),
                "dependents": random.randint(0, 4)
            })
        elif entity_type == "corporation":
            attributes.update({
                "industry": random.choice(["technology", "manufacturing", "retail", "services"]),
                "revenue": random.randint(100000, 10000000),
                "employees": random.randint(1, 1000),
                "structure_type": random.choice(["C-Corp", "S-Corp", "LLC"])
            })

        return attributes


class EventGenerator(BaseGenerator):
    """Generates synthetic tax events"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.event_templates = {
            "income": ["salary", "bonus", "dividend", "capital_gain", "rental_income"],
            "deduction": ["business_expense", "charitable_contribution", "mortgage_interest", "medical_expense"],
            "credit": ["child_tax_credit", "earned_income_credit", "education_credit"]
        }

    def generate(self, event_category: str = "income", entities: List[TaxEntity] = None, **kwargs) -> TaxEvent:
        """Generate a tax event"""
        import random

        event_id = str(uuid.uuid4())
        event_types = self.event_templates.get(event_category, ["generic_event"])
        event_type = random.choice(event_types)

        # Generate amount based on event type
        amount = self._generate_amount(event_type)

        # Select involved entities
        entities_involved = []
        if entities:
            num_entities = min(random.randint(1, 2), len(entities))
            entities_involved = [e.id for e in random.sample(entities, num_entities)]

        return TaxEvent(
            id=event_id,
            event_type=event_type,
            amount=amount,
            date="2024-01-01",  # Simplified for now
            description=f"Generated {event_type} event",
            entities_involved=entities_involved,
            tax_implications=self._generate_tax_implications(event_type, amount)
        )

    def _generate_amount(self, event_type: str) -> float:
        """Generate realistic amounts for different event types"""
        import random

        amount_ranges = {
            "salary": (30000, 200000),
            "bonus": (1000, 50000),
            "dividend": (100, 10000),
            "capital_gain": (500, 100000),
            "business_expense": (100, 25000),
            "charitable_contribution": (50, 5000),
            "mortgage_interest": (2000, 30000),
            "child_tax_credit": (2000, 2000),  # Fixed amount
        }

        min_amt, max_amt = amount_ranges.get(event_type, (100, 10000))
        return round(random.uniform(min_amt, max_amt), 2)

    def _generate_tax_implications(self, event_type: str, amount: float) -> Dict[str, Any]:
        """Generate tax implications for the event"""
        implications = {
            "taxable_amount": amount,
            "applicable_rate": 0.22,  # Simplified
            "deductible": "deduction" in event_type,
            "credit_eligible": "credit" in event_type
        }
        return implications


class NarrativeGenerator(BaseGenerator):
    """Generates natural language narratives for tax cases"""

    def generate(self, case: TaxLawCase, **kwargs) -> str:
        """Generate a narrative description of the tax case"""
        narrative_parts = []

        # Introduction
        narrative_parts.append(f"Consider the following tax scenario: {case.title}")

        # Entity descriptions
        for entity in case.entities:
            entity_desc = self._describe_entity(entity)
            narrative_parts.append(entity_desc)

        # Event descriptions
        for event in case.events:
            event_desc = self._describe_event(event, case.entities)
            narrative_parts.append(event_desc)

        # Question
        narrative_parts.append(
            "Given this information, determine the correct tax treatment and calculate any applicable tax liability.")

        return " ".join(narrative_parts)

    def _describe_entity(self, entity: TaxEntity) -> str:
        """Create natural language description of an entity"""
        if entity.entity_type == "individual":
            attrs = entity.attributes
            return f"{entity.name} is a {attrs.get('age', 'middle-aged')}-year-old {attrs.get('filing_status', 'individual')} with {attrs.get('dependents', 0)} dependents."
        elif entity.entity_type == "corporation":
            attrs = entity.attributes
            return f"{entity.name} is a {attrs.get('structure_type', 'corporation')} in the {attrs.get('industry', 'business')} industry with {attrs.get('employees', 'several')} employees."
        return f"{entity.name} is a {entity.entity_type}."

    def _describe_event(self, event: TaxEvent, entities: List[TaxEntity]) -> str:
        """Create natural language description of an event"""
        entity_names = [e.name for e in entities if e.id in event.entities_involved]
        entity_str = " and ".join(entity_names) if entity_names else "the taxpayer"

        if event.amount:
            return f"{entity_str} had {event.event_type.replace('_', ' ')} of ${event.amount:,.2f}."
        else:
            return f"{entity_str} experienced {event.event_type.replace('_', ' ')}."


class ReasoningChainGenerator(BaseGenerator):
    """Generates step-by-step reasoning chains for tax cases"""

    def generate(self, case: TaxLawCase, **kwargs) -> List[ReasoningStepData]:
        """Generate a complete reasoning chain"""
        reasoning_steps = []

        # Step 1: Identify relevant facts
        fact_step = self._create_fact_identification_step(case)
        reasoning_steps.append(fact_step)

        # Step 2: Apply tax rules
        rule_step = self._create_rule_application_step(case)
        reasoning_steps.append(rule_step)

        # Step 3: Perform calculations
        calc_step = self._create_calculation_step(case)
        reasoning_steps.append(calc_step)

        # Step 4: Draw conclusions
        conclusion_step = self._create_conclusion_step(case)
        reasoning_steps.append(conclusion_step)

        return reasoning_steps

    def _create_fact_identification_step(self, case: TaxLawCase) -> ReasoningStepData:
        """Create fact identification reasoning step"""
        facts = []
        for event in case.events:
            facts.append(f"{event.event_type}: ${event.amount}")

        return ReasoningStepData(
            step_id=str(uuid.uuid4()),
            step_type=ReasoningStep.FACT_IDENTIFICATION,
            description="Identify relevant tax facts from the scenario",
            input_data={"narrative": case.narrative},
            output_data={"identified_facts": facts},
            reasoning_text=f"From the scenario, I identify the following tax-relevant facts: {', '.join(facts)}"
        )

    def _create_rule_application_step(self, case: TaxLawCase) -> ReasoningStepData:
        """Create rule application reasoning step"""
        return ReasoningStepData(
            step_id=str(uuid.uuid4()),
            step_type=ReasoningStep.RULE_APPLICATION,
            description="Apply relevant tax rules to the identified facts",
            input_data={"facts": [e.event_type for e in case.events]},
            output_data={"applicable_rules": ["IRC Section 61", "IRC Section 162"]},
            reasoning_text="Applying IRC Section 61 for income recognition and IRC Section 162 for business deductions."
        )

    def _create_calculation_step(self, case: TaxLawCase) -> ReasoningStepData:
        """Create calculation reasoning step"""
        total_income = sum(e.amount for e in case.events if "income" in e.event_type)
        total_deductions = sum(e.amount for e in case.events if "deduction" in e.event_type)

        return ReasoningStepData(
            step_id=str(uuid.uuid4()),
            step_type=ReasoningStep.CALCULATION,
            description="Calculate tax liability based on applied rules",
            input_data={"income": total_income, "deductions": total_deductions},
            output_data={"taxable_income": total_income - total_deductions},
            reasoning_text=f"Total income: ${total_income}, Total deductions: ${total_deductions}, Taxable income: ${total_income - total_deductions}"
        )

    def _create_conclusion_step(self, case: TaxLawCase) -> ReasoningStepData:
        """Create conclusion reasoning step"""
        return ReasoningStepData(
            step_id=str(uuid.uuid4()),
            step_type=ReasoningStep.CONCLUSION,
            description="Draw final conclusion about tax treatment",
            input_data={"calculations": "completed"},
            output_data={"final_answer": case.ground_truth_answer},
            reasoning_text="Based on the calculations and rule applications, the final tax liability has been determined."
        )


class TaxLawCaseGenerator(BaseGenerator):
    """Main generator for complete tax law cases"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.entity_gen = EntityGenerator(config)
        self.event_gen = EventGenerator(config)
        self.narrative_gen = NarrativeGenerator(config)
        self.reasoning_gen = ReasoningChainGenerator(config)

    def generate(self, complexity_level: ComplexityLevel = ComplexityLevel.BASIC, **kwargs) -> TaxLawCase:
        """Generate a complete tax law case"""
        case_id = str(uuid.uuid4())

        # Generate entities based on complexity
        num_entities = self._get_entity_count_for_complexity(complexity_level)
        entities = [self.entity_gen.generate() for _ in range(num_entities)]

        # Generate events based on complexity
        num_events = self._get_event_count_for_complexity(complexity_level)
        events = []
        for _ in range(num_events):
            event_category = self._select_event_category()
            event = self.event_gen.generate(event_category, entities)
            events.append(event)

        # Create initial case structure
        case = TaxLawCase(
            case_id=case_id,
            title=f"Tax Case {case_id[:8]}",
            narrative="",  # Will be generated
            entities=entities,
            events=events,
            complexity_level=complexity_level,
            reasoning_chain=[],  # Will be generated
            ground_truth_answer="To be determined",
            metadata={"generated_at": "2024-09-02", "version": "1.0"}
        )

        # Generate narrative
        case.narrative = self.narrative_gen.generate(case)

        # Generate reasoning chain
        case.reasoning_chain = self.reasoning_gen.generate(case)

        # Calculate ground truth answer
        case.ground_truth_answer = self._calculate_ground_truth(case)

        logger.info(f"Generated tax case {case_id} with complexity {complexity_level.value}")
        return case

    def _get_entity_count_for_complexity(self, complexity: ComplexityLevel) -> int:
        """Determine number of entities based on complexity"""
        complexity_mapping = {
            ComplexityLevel.BASIC: 1,
            ComplexityLevel.INTERMEDIATE: 2,
            ComplexityLevel.ADVANCED: 3,
            ComplexityLevel.EXPERT: 4
        }
        return complexity_mapping.get(complexity, 1)

    def _get_event_count_for_complexity(self, complexity: ComplexityLevel) -> int:
        """Determine number of events based on complexity"""
        complexity_mapping = {
            ComplexityLevel.BASIC: 2,
            ComplexityLevel.INTERMEDIATE: 4,
            ComplexityLevel.ADVANCED: 6,
            ComplexityLevel.EXPERT: 8
        }
        return complexity_mapping.get(complexity, 2)

    def _select_event_category(self) -> str:
        """Select an event category for generation"""
        import random
        return random.choice(["income", "deduction", "credit"])

    def _calculate_ground_truth(self, case: TaxLawCase) -> str:
        """Calculate the ground truth answer for the case"""
        total_income = sum(e.amount for e in case.events if "income" in e.event_type)
        total_deductions = sum(e.amount for e in case.events if "deduction" in e.event_type)
        taxable_income = max(0, total_income - total_deductions)

        # Simplified tax calculation (22% rate)
        tax_liability = taxable_income * 0.22

        return f"The taxable income is ${taxable_income:,.2f} and the estimated tax liability is ${tax_liability:,.2f}."


# Usage Example and Testing
def main():
    """Example usage of the tax law case generator"""

    # Create generator
    generator = TaxLawCaseGenerator()

    # Generate cases of different complexity levels
    for complexity in ComplexityLevel:
        print(f"\n=== Generating {complexity.value.upper()} Case ===")
        case = generator.generate(complexity_level=complexity)

        print(f"Case ID: {case.case_id}")
        print(f"Title: {case.title}")
        print(f"Narrative: {case.narrative[:200]}...")
        print(f"Number of Entities: {len(case.entities)}")
        print(f"Number of Events: {len(case.events)}")
        print(f"Reasoning Steps: {len(case.reasoning_chain)}")
        print(f"Ground Truth: {case.ground_truth_answer}")


if __name__ == "__main__":
    main()