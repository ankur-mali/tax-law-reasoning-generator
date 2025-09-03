"""
Tax Law Reasoning Generator Package
Based on MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning

This package provides comprehensive tools for generating synthetic tax law cases
with AI-enhanced narratives and reasoning chains for evaluating GenAI capabilities.
"""

__version__ = "1.0.0"

# Core generator classes and enums
from .tax_law_generator import (
    TaxLawCaseGenerator,
    ComplexityLevel,
    TaxLawCase,
    TaxEntity,
    TaxEvent,
    ReasoningStepData,
    ReasoningStep,
    EntityGenerator,
    EventGenerator,
    NarrativeGenerator,
    ReasoningChainGenerator,
)

# Configuration and evaluation classes
from .config_evaluation import (
    GenerationConfig,
    ConfigManager,
    EvaluationMetrics,
    CaseEvaluator,
    DatasetGenerator,
    GenerativeAIIntegration as LegacyAIIntegration,  # Legacy import
)

# AI integration classes
from .ai_integration import (
    GenerativeAIIntegration,
    AIConfig,
    AIProvider,
    OpenAIProvider,
    EnhancedTaxLawCaseGenerator,
    create_openai_integration,
    setup_enhanced_generator,
)

# Utility imports for convenience
# from .utils import *

# Public API - what users should import
__all__ = [
    # Core generation
    "TaxLawCaseGenerator",
    "ComplexityLevel",
    "TaxLawCase",
    "TaxEntity",
    "TaxEvent",
    "ReasoningStepData",
    "ReasoningStep",

    # Specialized generators
    "EntityGenerator",
    "EventGenerator",
    "NarrativeGenerator",
    "ReasoningChainGenerator",

    # Configuration and evaluation
    "GenerationConfig",
    "ConfigManager",
    "EvaluationMetrics",
    "CaseEvaluator",
    "DatasetGenerator",

    # AI Integration
    "GenerativeAIIntegration",
    "AIConfig",
    "AIProvider",
    "OpenAIProvider",
    "EnhancedTaxLawCaseGenerator",

    # Utility functions
    "create_openai_integration",
    "setup_enhanced_generator",
]

# Package metadata
__author__ = "Ankur Mali"
__email__ = "ankurmali02@gmail.com"
__description__ = "AI-powered synthetic tax law reasoning data generator"
__license__ = "MIT"
__url__ = "https://github.com/ankur-mali/tax-law-reasoning-generator"

# Package-level configuration
DEFAULT_CONFIG = {
    "complexity_distribution": {
        "basic": 0.3,
        "intermediate": 0.4,
        "advanced": 0.2,
        "expert": 0.1
    },
    "ai_model": "gpt-4",
    "narrative_length_target": 500,
    "include_distractors": True,
    "min_reasoning_steps": 3,
    "max_reasoning_steps": 8
}

# Initialize logging for the package
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Prevent "no handler" warnings

# Package initialization message
def _initialize_package():
    """Initialize package-level settings"""
    logger.info(f"Tax Law Reasoning Generator v{__version__} initialized")
    logger.info("Based on MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning")

# Run initialization
_initialize_package()
