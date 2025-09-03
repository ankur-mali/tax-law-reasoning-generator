"""
AI Integration Module
Integrates OpenAI API with the Tax Law Reasoning Data Generator
Based on MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning
"""

import openai
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import os

from openai import OpenAI
import openai
# Import from main modules
from .tax_law_generator import TaxLawCase, ReasoningStepData, ReasoningStep

logger = logging.getLogger(__name__)


@dataclass
class AIConfig:
    """Configuration for AI integration"""

    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1500
    max_retries: int = 3
    retry_delay: float = 1.0

    # Enhancement-specific settings
    narrative_enhancement_temperature: float = 0.7
    reasoning_validation_temperature: float = 0.1
    reasoning_generation_temperature: float = 0.4


class AIProvider(ABC):
    """Abstract base class for AI providers"""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the AI provider"""
        pass

    @abstractmethod
    def validate_response(self, response: str) -> bool:
        """Validate AI response format"""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI API provider implementation"""

    def __init__(self, config: AIConfig):
        self.config = config

        # Set API key from config or environment
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key in config")

        # Initialize OpenAI client (v1.0.0+ style)
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI provider with model: {config.model}")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""

        # Override default config with kwargs
        temperature = kwargs.get('temperature', self.config.temperature)
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)

        for attempt in range(self.config.max_retries):
            try:
                # Updated API call for v1.0.0+
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                return response.choices[0].message.content.strip()

            except openai.RateLimitError as e:
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.config.max_retries}")
                time.sleep(wait_time)

            except openai.APIConnectionError as e:
                logger.error(f"OpenAI connection error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay)
            except openai.APIError as e:
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay)

        raise Exception(f"Failed to generate text after {self.config.max_retries} attempts")

    def validate_response(self, response: str) -> bool:
        """Validate OpenAI response format"""
        return bool(response and len(response.strip()) > 0)



class GenerativeAIIntegration:
    """
    Main AI integration class for enhancing tax law case generation
    Supports narrative enhancement, reasoning generation, and validation
    """

    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self.provider = self._initialize_provider()

        # Prompt templates for different tasks
        self.prompt_templates = {
            'narrative_enhancement': self._load_narrative_prompt_template(),
            'reasoning_generation': self._load_reasoning_prompt_template(),
            'reasoning_validation': self._load_validation_prompt_template(),
            'complexity_assessment': self._load_complexity_prompt_template()
        }

    def _initialize_provider(self) -> AIProvider:
        """Initialize the AI provider based on configuration"""
        if self.config.provider == "openai":
            return OpenAIProvider(self.config)
        else:
            raise ValueError(f"Unsupported AI provider: {self.config.provider}")

    def enhance_narrative(self, case: TaxLawCase, enhancement_level: str = "standard") -> str:
        """
        Enhance the narrative quality of a tax law case

        Args:
            case: The tax law case to enhance
            enhancement_level: Level of enhancement ('basic', 'standard', 'detailed')

        Returns:
            Enhanced narrative text
        """

        # Prepare context data for enhancement
        context = {
            'entities': [{'name': e.name, 'type': e.entity_type, 'attributes': e.attributes} for e in case.entities],
            'events': [{'type': e.event_type, 'amount': e.amount, 'description': e.description} for e in case.events],
            'complexity_level': case.complexity_level.value,
            'original_narrative': case.narrative,
            'enhancement_level': enhancement_level
        }

        prompt = self.prompt_templates['narrative_enhancement'].format(**context)

        try:
            enhanced_narrative = self.provider.generate_text(
                prompt,
                temperature=self.config.narrative_enhancement_temperature,
                max_tokens=800
            )

            logger.info(f"Successfully enhanced narrative for case {case.case_id}")
            return enhanced_narrative

        except Exception as e:
            logger.error(f"Failed to enhance narrative for case {case.case_id}: {e}")
            return case.narrative  # Return original on failure

    def generate_advanced_reasoning(self, case: TaxLawCase) -> List[ReasoningStepData]:
        """
        Generate sophisticated reasoning chains using AI

        Args:
            case: The tax law case to generate reasoning for

        Returns:
            List of enhanced reasoning steps
        """

        # Prepare context for reasoning generation
        context = {
            'narrative': case.narrative,
            'entities': [{'name': e.name, 'type': e.entity_type, 'attributes': e.attributes} for e in case.entities],
            'events': [{'type': e.event_type, 'amount': e.amount, 'implications': e.tax_implications} for e in
                       case.events],
            'complexity_level': case.complexity_level.value,
            'target_steps': len(case.reasoning_chain)
        }

        prompt = self.prompt_templates['reasoning_generation'].format(**context)

        try:
            reasoning_response = self.provider.generate_text(
                prompt,
                temperature=self.config.reasoning_generation_temperature,
                max_tokens=1200
            )

            # Parse AI response into structured reasoning steps
            enhanced_steps = self._parse_reasoning_response(reasoning_response, case)

            logger.info(f"Generated {len(enhanced_steps)} enhanced reasoning steps for case {case.case_id}")
            return enhanced_steps

        except Exception as e:
            logger.error(f"Failed to generate advanced reasoning for case {case.case_id}: {e}")
            return case.reasoning_chain  # Return original on failure

    def validate_reasoning_chain(self, case: TaxLawCase) -> Dict[str, Any]:
        """
        Validate the reasoning chain for accuracy and completeness

        Args:
            case: The tax law case to validate

        Returns:
            Validation results with scores and feedback
        """

        reasoning_text = "\n".join([
            f"Step {i + 1} ({step.step_type.value}): {step.reasoning_text}"
            for i, step in enumerate(case.reasoning_chain)
        ])

        context = {
            'narrative': case.narrative,
            'reasoning_chain': reasoning_text,
            'ground_truth': case.ground_truth_answer,
            'complexity_level': case.complexity_level.value
        }

        prompt = self.prompt_templates['reasoning_validation'].format(**context)

        try:
            validation_response = self.provider.generate_text(
                prompt,
                temperature=self.config.reasoning_validation_temperature,
                max_tokens=600
            )

            # Parse validation response
            validation_result = self._parse_validation_response(validation_response)
            validation_result['case_id'] = case.case_id

            logger.info(f"Validated reasoning chain for case {case.case_id}")
            return validation_result

        except Exception as e:
            logger.error(f"Failed to validate reasoning for case {case.case_id}: {e}")
            return {
                'case_id': case.case_id,
                'is_logically_sound': True,
                'tax_application_errors': [],
                'missing_steps': [],
                'overall_confidence_score': 0.5,
                'feedback': f"Validation failed: {e}"
            }

    def assess_case_complexity(self, case: TaxLawCase) -> Dict[str, float]:
        """
        Use AI to assess the true complexity of a generated case

        Args:
            case: The tax law case to assess

        Returns:
            Complexity assessment scores
        """

        context = {
            'narrative': case.narrative,
            'num_entities': len(case.entities),
            'num_events': len(case.events),
            'num_reasoning_steps': len(case.reasoning_chain),
            'stated_complexity': case.complexity_level.value
        }

        prompt = self.prompt_templates['complexity_assessment'].format(**context)

        try:
            complexity_response = self.provider.generate_text(
                prompt,
                temperature=0.1,
                max_tokens=400
            )

            complexity_scores = self._parse_complexity_response(complexity_response)

            logger.info(f"Assessed complexity for case {case.case_id}")
            return complexity_scores

        except Exception as e:
            logger.error(f"Failed to assess complexity for case {case.case_id}: {e}")
            return {
                'cognitive_load': 0.5,
                'legal_complexity': 0.5,
                'calculation_difficulty': 0.5,
                'overall_complexity': 0.5
            }

    def _parse_reasoning_response(self, response: str, case: TaxLawCase) -> List[ReasoningStepData]:
        """Parse AI-generated reasoning response into structured steps"""

        enhanced_steps = []
        lines = response.split('\n')
        current_step = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for step markers (e.g., "Step 1:", "1.", etc.)
            if any(marker in line.lower() for marker in ['step ', 'phase ', 'stage ']):
                if current_step:
                    enhanced_steps.append(current_step)

                # Create new reasoning step
                step_id = f"ai_enhanced_{len(enhanced_steps) + 1}"
                step_type = self._infer_step_type(line)

                current_step = ReasoningStepData(
                    step_id=step_id,
                    step_type=step_type,
                    description=line,
                    input_data={},
                    output_data={},
                    reasoning_text=line
                )
            elif current_step:
                # Append to current step's reasoning text
                current_step.reasoning_text += f" {line}"

        # Add final step
        if current_step:
            enhanced_steps.append(current_step)

        # If no steps were parsed, return original
        return enhanced_steps if enhanced_steps else case.reasoning_chain

    def _infer_step_type(self, step_text: str) -> ReasoningStep:
        """Infer the reasoning step type from text content"""

        text_lower = step_text.lower()

        if any(keyword in text_lower for keyword in ['identify', 'fact', 'given', 'information']):
            return ReasoningStep.FACT_IDENTIFICATION
        elif any(keyword in text_lower for keyword in ['rule', 'law', 'code', 'regulation', 'apply']):
            return ReasoningStep.RULE_APPLICATION
        elif any(keyword in text_lower for keyword in ['calculate', 'compute', 'math', 'total', 'sum']):
            return ReasoningStep.CALCULATION
        elif any(keyword in text_lower for keyword in ['interpret', 'analyze', 'consider', 'determine']):
            return ReasoningStep.INTERPRETATION
        elif any(keyword in text_lower for keyword in ['conclude', 'therefore', 'result', 'final']):
            return ReasoningStep.CONCLUSION
        else:
            return ReasoningStep.INTERPRETATION  # Default

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse AI validation response into structured results"""

        try:
            # Try to parse as JSON first
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: extract information from text
            validation_result = {
                'is_logically_sound': True,
                'tax_application_errors': [],
                'missing_steps': [],
                'overall_confidence_score': 0.7,
                'feedback': response
            }

            # Simple text analysis for validation indicators
            response_lower = response.lower()

            if any(indicator in response_lower for indicator in ['incorrect', 'wrong', 'error', 'invalid']):
                validation_result['is_logically_sound'] = False
                validation_result['overall_confidence_score'] = 0.3

            if any(indicator in response_lower for indicator in ['missing', 'incomplete', 'lacking']):
                validation_result['missing_steps'] = ['AI identified missing steps']
                validation_result['overall_confidence_score'] = min(validation_result['overall_confidence_score'], 0.5)

            return validation_result

    def _parse_complexity_response(self, response: str) -> Dict[str, float]:
        """Parse AI complexity assessment response"""

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: default scores
            return {
                'cognitive_load': 0.5,
                'legal_complexity': 0.5,
                'calculation_difficulty': 0.5,
                'overall_complexity': 0.5
            }

    def _load_narrative_prompt_template(self) -> str:
        """Load prompt template for narrative enhancement"""

        return """
You are an expert tax attorney and technical writer. Please enhance this tax law case narrative to make it more realistic, engaging, and professional while maintaining complete factual accuracy.

Original Case Details:
- Entities: {entities}
- Events: {events}
- Complexity Level: {complexity_level}
- Enhancement Level: {enhancement_level}

Current Narrative:
{original_narrative}

Requirements:
1. Keep ALL factual information (names, amounts, dates, relationships) exactly the same
2. Improve language flow and professional tone
3. Add realistic contextual details that don't affect tax implications
4. Ensure the narrative reads like a real tax consultation scenario
5. Maintain appropriate complexity for the {complexity_level} level
6. Length should be appropriate for {enhancement_level} enhancement (300-800 words)

Enhanced Narrative:
"""

    def _load_reasoning_prompt_template(self) -> str:
        """Load prompt template for reasoning generation"""

        return """
You are an expert tax attorney creating a detailed chain-of-thought reasoning for a complex tax case. Generate a comprehensive, step-by-step analysis.

Case Narrative:
{narrative}

Case Details:
- Entities: {entities}
- Events: {events}
- Complexity Level: {complexity_level}
- Target Number of Steps: {target_steps}

Generate a detailed reasoning chain that includes:
1. Fact Identification: Extract and organize relevant tax facts
2. Rule Application: Identify and apply relevant IRC sections and tax principles
3. Calculations: Show detailed mathematical computations with explanations
4. Interpretation: Analyze complex scenarios and edge cases
5. Conclusion: Synthesize findings into final determination

Requirements:
- Each step should be clearly numbered and labeled by type
- Include specific IRC citations where applicable
- Show all calculation work with explanations
- Address potential complications or alternative interpretations
- Use professional tax terminology
- Ensure logical flow between steps

Reasoning Chain:
"""

    def _load_validation_prompt_template(self) -> str:
        """Load prompt template for reasoning validation"""

        return """
You are an expert tax law reviewer validating a reasoning chain for accuracy and completeness. Provide a thorough assessment in JSON format.

Case Narrative:
{narrative}

Reasoning Chain to Validate:
{reasoning_chain}

Ground Truth Answer:
{ground_truth}

Case Complexity: {complexity_level}

Please analyze the reasoning chain and respond with JSON containing:
{{
    "is_logically_sound": boolean,
    "tax_application_errors": [list of specific errors found],
    "missing_steps": [list of important missing reasoning steps],
    "overall_confidence_score": float (0.0 to 1.0),
    "strengths": [list of strong aspects],
    "improvement_suggestions": [list of specific improvements],
    "technical_accuracy_score": float (0.0 to 1.0),
    "completeness_score": float (0.0 to 1.0)
}}

Validation Result:
"""

    def _load_complexity_prompt_template(self) -> str:
        """Load prompt template for complexity assessment"""

        return """
You are an expert in tax law complexity assessment. Analyze this case and provide complexity scores in JSON format.

Case Narrative:
{narrative}

Case Statistics:
- Number of Entities: {num_entities}
- Number of Events: {num_events}
- Number of Reasoning Steps: {num_reasoning_steps}
- Stated Complexity Level: {stated_complexity}

Assess complexity across these dimensions (0.0 to 1.0 scale):

{{
    "cognitive_load": float (mental effort required to follow the case),
    "legal_complexity": float (sophistication of tax law concepts involved),
    "calculation_difficulty": float (mathematical computation complexity),
    "interaction_complexity": float (complexity of entity/event relationships),
    "overall_complexity": float (综合复杂度评估),
    "appropriate_for_level": boolean (is complexity appropriate for stated level?)
}}

Complexity Assessment:
"""


# Integration with main generator
class EnhancedTaxLawCaseGenerator:
    """
    Enhanced version of TaxLawCaseGenerator that integrates AI capabilities
    """

    def __init__(self, base_generator, ai_integration: GenerativeAIIntegration = None):
        self.base_generator = base_generator
        self.ai_integration = ai_integration

    def generate(self, complexity_level, use_ai_enhancement: bool = True, **kwargs) -> TaxLawCase:
        """Generate a case with optional AI enhancement"""

        # Generate base case using original generator
        case = self.base_generator.generate(complexity_level=complexity_level, **kwargs)

        if use_ai_enhancement and self.ai_integration:
            # Enhance narrative
            case.narrative = self.ai_integration.enhance_narrative(case)

            # Generate advanced reasoning
            case.reasoning_chain = self.ai_integration.generate_advanced_reasoning(case)

            # Validate and potentially adjust
            validation = self.ai_integration.validate_reasoning_chain(case)
            if validation.get('overall_confidence_score', 1.0) < 0.6:
                logger.warning(f"Low confidence validation for case {case.case_id}: {validation.get('feedback', '')}")

        return case


# Utility functions for easy setup
def create_openai_integration(api_key: str = None, model: str = "gpt-4") -> GenerativeAIIntegration:
    """Create an OpenAI integration instance with sensible defaults"""

    config = AIConfig(
        provider="openai",
        model=model,
        api_key=api_key,
        temperature=0.3,
        max_tokens=1500
    )

    return GenerativeAIIntegration(config)


def setup_enhanced_generator(base_generator, openai_api_key: str = None):
    """Set up an enhanced generator with AI integration"""

    ai_integration = create_openai_integration(api_key=openai_api_key)
    return EnhancedTaxLawCaseGenerator(base_generator, ai_integration)


