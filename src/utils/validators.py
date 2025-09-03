"""
Validation Module for Tax Law Reasoning Generator
Provides comprehensive validation for entities, events, and cases to ensure data quality and legal accuracy
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import re
from datetime import datetime, date
from decimal import Decimal, InvalidOperation

from .data_structures import (
    TaxLawCase,
    TaxEntity,
    TaxEvent,
    ReasoningStepData,
    ComplexityLevel,
    ReasoningStep,
    EntityType,
    EventCategory
)


class ValidationError(Exception):
    """Custom exception for validation errors"""

    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class EntityValidator:
    """Validator for tax entities with domain-specific rules"""

    VALID_ENTITY_TYPES = {e.value for e in EntityType}
    REQUIRED_INDIVIDUAL_ATTRS = {"age", "filing_status", "income_level"}
    REQUIRED_CORPORATION_ATTRS = {"industry", "revenue", "employees", "structure_type"}
    VALID_FILING_STATUS = {"single", "married_joint", "married_separate", "head_of_household"}
    VALID_INCOME_LEVELS = {"low", "medium", "high", "ultra_high"}
    VALID_CORP_STRUCTURES = {"c_corporation", "s_corporation", "llc", "partnership"}

    @classmethod
    def validate_entity(cls, entity: TaxEntity) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of tax entity
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        # Basic field validation
        if not entity.id or not isinstance(entity.id, str) or len(entity.id.strip()) == 0:
            errors.append("Entity ID is required and must be non-empty string")

        if not entity.name or not isinstance(entity.name, str) or len(entity.name.strip()) == 0:
            errors.append("Entity name is required and must be non-empty string")

        if not entity.entity_type or entity.entity_type not in cls.VALID_ENTITY_TYPES:
            errors.append(f"Entity type must be one of: {cls.VALID_ENTITY_TYPES}")

        # Entity-specific validation
        if entity.entity_type == EntityType.INDIVIDUAL.value:
            errors.extend(cls._validate_individual(entity))
        elif entity.entity_type == EntityType.CORPORATION.value:
            errors.extend(cls._validate_corporation(entity))
        elif entity.entity_type == EntityType.PARTNERSHIP.value:
            errors.extend(cls._validate_partnership(entity))
        elif entity.entity_type == EntityType.TRUST.value:
            errors.extend(cls._validate_trust(entity))

        return len(errors) == 0, errors

    @classmethod
    def _validate_individual(cls, entity: TaxEntity) -> List[str]:
        """Validate individual taxpayer specific attributes"""
        errors = []
        attrs = entity.attributes

        # Check required attributes
        missing_attrs = cls.REQUIRED_INDIVIDUAL_ATTRS - set(attrs.keys())
        if missing_attrs:
            errors.append(f"Individual missing required attributes: {missing_attrs}")

        # Age validation
        if "age" in attrs:
            age = attrs["age"]
            if not isinstance(age, int) or age < 0 or age > 120:
                errors.append("Age must be integer between 0 and 120")

        # Filing status validation
        if "filing_status" in attrs:
            filing_status = attrs["filing_status"]
            if filing_status not in cls.VALID_FILING_STATUS:
                errors.append(f"Filing status must be one of: {cls.VALID_FILING_STATUS}")

        # Income level validation
        if "income_level" in attrs:
            income_level = attrs["income_level"]
            if income_level not in cls.VALID_INCOME_LEVELS:
                errors.append(f"Income level must be one of: {cls.VALID_INCOME_LEVELS}")

        # Dependents validation
        if "dependents" in attrs:
            dependents = attrs["dependents"]
            if not isinstance(dependents, int) or dependents < 0 or dependents > 10:
                errors.append("Dependents must be integer between 0 and 10")

        # Cross-validation: head of household must have dependents
        if (attrs.get("filing_status") == "head_of_household" and
                attrs.get("dependents", 0) == 0):
            errors.append("Head of household must have at least one dependent")

        return errors

    @classmethod
    def _validate_corporation(cls, entity: TaxEntity) -> List[str]:
        """Validate corporation specific attributes"""
        errors = []
        attrs = entity.attributes

        # Check required attributes
        missing_attrs = cls.REQUIRED_CORPORATION_ATTRS - set(attrs.keys())
        if missing_attrs:
            errors.append(f"Corporation missing required attributes: {missing_attrs}")

        # Revenue validation
        if "revenue" in attrs:
            revenue = attrs["revenue"]
            if not isinstance(revenue, (int, float)) or revenue < 0:
                errors.append("Revenue must be non-negative number")
            if revenue > 1000000000000:  # $1 trillion cap for sanity
                errors.append("Revenue exceeds reasonable maximum ($1T)")

        # Employee count validation
        if "employees" in attrs:
            employees = attrs["employees"]
            if not isinstance(employees, int) or employees < 0:
                errors.append("Employee count must be non-negative integer")
            if employees > 10000000:  # 10M employees cap for sanity
                errors.append("Employee count exceeds reasonable maximum (10M)")

        # Structure type validation
        if "structure_type" in attrs:
            structure = attrs["structure_type"]
            if structure not in cls.VALID_CORP_STRUCTURES:
                errors.append(f"Structure type must be one of: {cls.VALID_CORP_STRUCTURES}")

        # Industry validation
        if "industry" in attrs:
            industry = attrs["industry"]
            if not isinstance(industry, str) or len(industry.strip()) == 0:
                errors.append("Industry must be non-empty string")

        return errors

    @classmethod
    def _validate_partnership(cls, entity: TaxEntity) -> List[str]:
        """Validate partnership specific attributes"""
        errors = []
        attrs = entity.attributes

        # Partner count validation
        if "partner_count" in attrs:
            count = attrs["partner_count"]
            if not isinstance(count, int) or count < 2:
                errors.append("Partnership must have at least 2 partners")
            if count > 1000:
                errors.append("Partnership partner count exceeds reasonable maximum (1000)")

        return errors

    @classmethod
    def _validate_trust(cls, entity: TaxEntity) -> List[str]:
        """Validate trust specific attributes"""
        errors = []
        attrs = entity.attributes

        # Trust type validation
        valid_trust_types = {"revocable", "irrevocable", "charitable", "grantor"}
        if "trust_type" in attrs:
            trust_type = attrs["trust_type"]
            if trust_type not in valid_trust_types:
                errors.append(f"Trust type must be one of: {valid_trust_types}")

        # Beneficiaries validation
        if "beneficiaries" in attrs:
            beneficiaries = attrs["beneficiaries"]
            if not isinstance(beneficiaries, list) or len(beneficiaries) == 0:
                errors.append("Trust must have at least one beneficiary")

        return errors


class EventValidator:
    """Validator for tax events with tax law compliance checks"""

    INCOME_EVENT_TYPES = {
        "salary", "wages", "bonus", "commission", "tips", "dividend", "interest",
        "capital_gain", "rental_income", "business_income", "pension", "social_security",
        "unemployment", "gambling_winnings", "prize_winnings", "stock_compensation"
    }

    DEDUCTION_EVENT_TYPES = {
        "mortgage_interest", "property_tax", "state_income_tax", "charitable_contribution",
        "medical_expenses", "business_expenses", "home_office", "depreciation",
        "employee_expenses", "investment_expenses", "legal_fees", "tax_preparation"
    }

    CREDIT_EVENT_TYPES = {
        "child_tax_credit", "earned_income_credit", "education_credit", "child_care_credit",
        "retirement_savings_credit", "adoption_credit", "foreign_tax_credit"
    }

    ALL_EVENT_TYPES = INCOME_EVENT_TYPES | DEDUCTION_EVENT_TYPES | CREDIT_EVENT_TYPES

    @classmethod
    def validate_event(cls, event: TaxEvent, entity_ids: List[str]) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of tax event
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        # Basic field validation
        if not event.id or not isinstance(event.id, str):
            errors.append("Event ID is required and must be string")

        if not event.event_type or event.event_type not in cls.ALL_EVENT_TYPES:
            errors.append(f"Event type must be one of: {sorted(cls.ALL_EVENT_TYPES)}")

        # Amount validation
        if event.amount is not None:
            if not isinstance(event.amount, (int, float, Decimal)):
                errors.append("Event amount must be numeric")
            elif event.amount < 0 and event.event_type not in {"capital_loss", "net_operating_loss"}:
                errors.append(f"Amount cannot be negative for event type: {event.event_type}")
            elif event.amount > 1000000000:  # $1B sanity check
                errors.append("Event amount exceeds reasonable maximum ($1B)")

        # Date validation
        if event.date:
            if not cls._validate_date_format(event.date):
                errors.append("Date must be in YYYY-MM-DD format")
            elif not cls._validate_tax_year(event.date):
                errors.append("Date must be within reasonable tax years (1900-2030)")

        # Entity involvement validation
        if event.entities_involved:
            for entity_id in event.entities_involved:
                if entity_id not in entity_ids:
                    errors.append(f"Referenced entity ID not found: {entity_id}")
        else:
            errors.append("Event must involve at least one entity")

        # Event-specific validation
        if event.event_type in cls.INCOME_EVENT_TYPES:
            errors.extend(cls._validate_income_event(event))
        elif event.event_type in cls.DEDUCTION_EVENT_TYPES:
            errors.extend(cls._validate_deduction_event(event))
        elif event.event_type in cls.CREDIT_EVENT_TYPES:
            errors.extend(cls._validate_credit_event(event))

        return len(errors) == 0, errors

    @classmethod
    def _validate_date_format(cls, date_str: str) -> bool:
        """Validate date format YYYY-MM-DD"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    @classmethod
    def _validate_tax_year(cls, date_str: str) -> bool:
        """Validate date is within reasonable tax years"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return 1900 <= date_obj.year <= 2030
        except ValueError:
            return False

    @classmethod
    def _validate_income_event(cls, event: TaxEvent) -> List[str]:
        """Validate income-specific rules"""
        errors = []

        if event.amount is None:
            errors.append("Income events must have an amount")
        elif event.amount <= 0:
            errors.append("Income amounts must be positive")

        # Specific income type validations
        if event.event_type == "salary" and event.amount and event.amount > 50000000:
            errors.append("Salary amount exceeds reasonable maximum ($50M)")

        if event.event_type == "social_security" and event.amount and event.amount > 100000:
            errors.append("Social Security income exceeds legal maximum")

        return errors

    @classmethod
    def _validate_deduction_event(cls, event: TaxEvent) -> List[str]:
        """Validate deduction-specific rules"""
        errors = []

        if event.amount is None:
            errors.append("Deduction events must have an amount")
        elif event.amount <= 0:
            errors.append("Deduction amounts must be positive")

        # Specific deduction validations
        if event.event_type == "charitable_contribution":
            # Basic AGI limit check (simplified)
            if event.amount and event.amount > 10000000:  # $10M sanity check
                errors.append("Charitable contribution exceeds reasonable maximum")

        return errors

    @classmethod
    def _validate_credit_event(cls, event: TaxEvent) -> List[str]:
        """Validate credit-specific rules"""
        errors = []

        if event.amount is None:
            errors.append("Credit events must have an amount")
        elif event.amount <= 0:
            errors.append("Credit amounts must be positive")

        # Specific credit validations
        if event.event_type == "child_tax_credit" and event.amount:
            if event.amount > 3000:  # Current max per child
                errors.append("Child tax credit exceeds maximum amount")

        return errors


class ReasoningValidator:
    """Validator for reasoning chains with logical flow validation"""

    REQUIRED_STEP_TYPES = {
        ReasoningStep.FACT_IDENTIFICATION,
        ReasoningStep.RULE_APPLICATION,
        ReasoningStep.CALCULATION,
        ReasoningStep.CONCLUSION
    }

    @classmethod
    def validate_reasoning_chain(cls, chain: List[ReasoningStepData]) -> Tuple[bool, List[str]]:
        """
        Validate reasoning chain for logical flow and completeness
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        if not chain:
            errors.append("Reasoning chain cannot be empty")
            return False, errors

        # Check for required step types
        present_steps = {step.step_type for step in chain}
        missing_steps = cls.REQUIRED_STEP_TYPES - present_steps
        if missing_steps:
            missing_names = [step.value for step in missing_steps]
            errors.append(f"Missing required reasoning steps: {missing_names}")

        # Validate individual steps
        for i, step in enumerate(chain):
            step_errors = cls._validate_reasoning_step(step, i)
            errors.extend(step_errors)

        # Validate logical flow
        flow_errors = cls._validate_reasoning_flow(chain)
        errors.extend(flow_errors)

        return len(errors) == 0, errors

    @classmethod
    def _validate_reasoning_step(cls, step: ReasoningStepData, index: int) -> List[str]:
        """Validate individual reasoning step"""
        errors = []

        if not step.step_id:
            errors.append(f"Step {index}: Missing step ID")

        if not step.description:
            errors.append(f"Step {index}: Missing description")

        if not step.reasoning_text:
            errors.append(f"Step {index}: Missing reasoning text")
        elif len(step.reasoning_text.strip()) < 10:
            errors.append(f"Step {index}: Reasoning text too short")

        if step.confidence_score is not None:
            if not isinstance(step.confidence_score, (int, float)):
                errors.append(f"Step {index}: Confidence score must be numeric")
            elif not 0.0 <= step.confidence_score <= 1.0:
                errors.append(f"Step {index}: Confidence score must be between 0.0 and 1.0")

        return errors

    @classmethod
    def _validate_reasoning_flow(cls, chain: List[ReasoningStepData]) -> List[str]:
        """Validate logical flow of reasoning steps"""
        errors = []

        # Fact identification should come first
        first_step = chain[0]
        if first_step.step_type != ReasoningStep.FACT_IDENTIFICATION:
            errors.append("Reasoning should start with fact identification")

        # Conclusion should come last
        last_step = chain[-1]
        if last_step.step_type != ReasoningStep.CONCLUSION:
            errors.append("Reasoning should end with conclusion")

        # Rule application should come before calculation
        rule_index = None
        calc_index = None
        for i, step in enumerate(chain):
            if step.step_type == ReasoningStep.RULE_APPLICATION:
                rule_index = i
            elif step.step_type == ReasoningStep.CALCULATION:
                calc_index = i

        if rule_index is not None and calc_index is not None and rule_index > calc_index:
            errors.append("Rule application should precede calculation")

        return errors


class CaseValidator:
    """Comprehensive validator for complete tax law cases"""

    @classmethod
    def validate_tax_case(cls, case: TaxLawCase) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of tax law case
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        # Basic case validation
        if not case.case_id:
            errors.append("Case ID is required")

        if not case.title:
            errors.append("Case title is required")

        if not case.narrative or len(case.narrative.strip()) < 50:
            errors.append("Case narrative is required and must be at least 50 characters")

        if not case.ground_truth_answer:
            errors.append("Ground truth answer is required")

        # Validate entities
        if not case.entities:
            errors.append("Case must have at least one entity")
        else:
            entity_ids = []
            for i, entity in enumerate(case.entities):
                is_valid, entity_errors = EntityValidator.validate_entity(entity)
                if not is_valid:
                    errors.extend([f"Entity {i}: {error}" for error in entity_errors])
                else:
                    entity_ids.append(entity.id)

        # Validate events
        if not case.events:
            errors.append("Case must have at least one event")
        else:
            entity_ids = [e.id for e in case.entities]
            for i, event in enumerate(case.events):
                is_valid, event_errors = EventValidator.validate_event(event, entity_ids)
                if not is_valid:
                    errors.extend([f"Event {i}: {error}" for error in event_errors])

        # Validate reasoning chain
        if case.reasoning_chain:
            is_valid, reasoning_errors = ReasoningValidator.validate_reasoning_chain(case.reasoning_chain)
            if not is_valid:
                errors.extend([f"Reasoning: {error}" for error in reasoning_errors])
        else:
            errors.append("Case must have a reasoning chain")

        # Cross-validation: narrative should mention entities and events
        narrative_lower = case.narrative.lower()

        # Check if major entities are mentioned
        entity_mentions = 0
        for entity in case.entities:
            if entity.name.lower() in narrative_lower:
                entity_mentions += 1

        if entity_mentions < len(case.entities) / 2:
            errors.append("Narrative should mention most entities")

        # Check if major events are represented
        event_mentions = 0
        for event in case.events:
            event_type_words = event.event_type.replace("_", " ").split()
            if any(word in narrative_lower for word in event_type_words):
                event_mentions += 1

        if event_mentions < len(case.events) / 2:
            errors.append("Narrative should reference most events")

        # Complexity validation
        complexity_errors = cls._validate_case_complexity(case)
        errors.extend(complexity_errors)

        return len(errors) == 0, errors

    @classmethod
    def _validate_case_complexity(cls, case: TaxLawCase) -> List[str]:
        """Validate case complexity matches its content"""
        errors = []

        complexity = case.complexity_level
        num_entities = len(case.entities)
        num_events = len(case.events)
        num_reasoning_steps = len(case.reasoning_chain)

        # Complexity expectations
        complexity_rules = {
            ComplexityLevel.BASIC: {"max_entities": 2, "max_events": 4, "max_reasoning": 6},
            ComplexityLevel.INTERMEDIATE: {"max_entities": 3, "max_events": 6, "max_reasoning": 8},
            ComplexityLevel.ADVANCED: {"max_entities": 4, "max_events": 8, "max_reasoning": 10},
            ComplexityLevel.EXPERT: {"max_entities": 10, "max_events": 15, "max_reasoning": 15}
        }

        if complexity in complexity_rules:
            rules = complexity_rules[complexity]

            if complexity != ComplexityLevel.EXPERT:  # Expert can exceed limits
                if num_entities > rules["max_entities"]:
                    errors.append(f"Too many entities ({num_entities}) for {complexity.value} complexity")

                if num_events > rules["max_events"]:
                    errors.append(f"Too many events ({num_events}) for {complexity.value} complexity")

                if num_reasoning_steps > rules["max_reasoning"]:
                    errors.append(f"Too many reasoning steps ({num_reasoning_steps}) for {complexity.value} complexity")

        return errors


# Convenience functions
def validate_entity(entity: TaxEntity) -> bool:
    """Simple entity validation returning boolean"""
    is_valid, _ = EntityValidator.validate_entity(entity)
    return is_valid


def validate_event(event: TaxEvent, entity_ids: List[str]) -> bool:
    """Simple event validation returning boolean"""
    is_valid, _ = EventValidator.validate_event(event, entity_ids)
    return is_valid


def validate_reasoning_chain(chain: List[ReasoningStepData]) -> bool:
    """Simple reasoning chain validation returning boolean"""
    is_valid, _ = ReasoningValidator.validate_reasoning_chain(chain)
    return is_valid


def validate_tax_case(case: TaxLawCase) -> bool:
    """Simple case validation returning boolean"""
    is_valid, _ = CaseValidator.validate_tax_case(case)
    return is_valid


# Detailed validation functions
def validate_entity_detailed(entity: TaxEntity) -> Tuple[bool, List[str]]:
    """Detailed entity validation with error messages"""
    return EntityValidator.validate_entity(entity)


def validate_event_detailed(event: TaxEvent, entity_ids: List[str]) -> Tuple[bool, List[str]]:
    """Detailed event validation with error messages"""
    return EventValidator.validate_event(event, entity_ids)


def validate_reasoning_chain_detailed(chain: List[ReasoningStepData]) -> Tuple[bool, List[str]]:
    """Detailed reasoning chain validation with error messages"""
    return ReasoningValidator.validate_reasoning_chain(chain)


def validate_tax_case_detailed(case: TaxLawCase) -> Tuple[bool, List[str]]:
    """Detailed case validation with error messages"""
    return CaseValidator.validate_tax_case(case)
