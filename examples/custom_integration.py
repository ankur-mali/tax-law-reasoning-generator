"""
Custom AI Integration Example for Tax Law Reasoning Data Generator
Demonstrates advanced AI integration patterns, custom providers, and enhanced generation workflows
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax_law_generator.ai_integration import (
    GenerativeAIIntegration,
    AIConfig,
    AIProvider,
    OpenAIProvider,
    EnhancedTaxLawCaseGenerator
)
from src.tax_law_generator.tax_law_generator import (
    TaxLawCaseGenerator,
    ComplexityLevel
)
from src.tax_law_generator.config_evaluation import GenerationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomAIProvider(AIProvider):
    """
    Example custom AI provider implementation
    This could integrate with Claude, Gemini, or other AI services
    """

    def __init__(self, config: AIConfig, provider_name: str = "custom"):
        self.config = config
        self.provider_name = provider_name
        logger.info(f"Initialized {provider_name} AI provider")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Custom text generation implementation
        In a real scenario, this would call your preferred AI service
        """
        # Simulate custom AI processing
        logger.info(f"Generating text with {self.provider_name} provider...")

        # This is a mock implementation - replace with actual AI service calls
        enhanced_text = f"[Enhanced by {self.provider_name}] " + prompt[:200] + "..."

        return enhanced_text

    def validate_response(self, response: str) -> bool:
        """Validate response from custom provider"""
        return bool(response and len(response.strip()) > 0)


class MultiProviderAIIntegration(GenerativeAIIntegration):
    """
    Enhanced AI integration supporting multiple providers with fallback
    """

    def __init__(self, primary_config: AIConfig, fallback_config: Optional[AIConfig] = None):
        # Initialize primary provider
        super().__init__(primary_config)
        self.primary_provider = self.provider
        self.primary_config = primary_config

        # Initialize fallback provider if provided
        self.fallback_provider = None
        if fallback_config:
            if fallback_config.provider == "openai":
                self.fallback_provider = OpenAIProvider(fallback_config)
            elif fallback_config.provider == "custom":
                self.fallback_provider = CustomAIProvider(fallback_config, "fallback")

        logger.info(f"Multi-provider AI integration initialized")
        logger.info(f"Primary: {primary_config.provider}")
        if self.fallback_provider:
            logger.info(f"Fallback: {fallback_config.provider}")

    def generate_text_with_fallback(self, prompt: str, **kwargs) -> str:
        """Generate text with automatic fallback on failure"""

        try:
            # Try primary provider first
            result = self.primary_provider.generate_text(prompt, **kwargs)
            logger.info(f"✓ Primary provider ({self.primary_config.provider}) successful")
            return result

        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")

            if self.fallback_provider:
                try:
                    logger.info("Attempting fallback provider...")
                    result = self.fallback_provider.generate_text(prompt, **kwargs)
                    logger.info("✓ Fallback provider successful")
                    return result
                except Exception as e2:
                    logger.error(f"Fallback provider also failed: {e2}")

            # If all providers fail, return a basic response
            logger.error("All AI providers failed, returning basic response")
            return "AI enhancement unavailable - using base content"

    def enhance_narrative_with_style(self, case, style: str = "professional") -> str:
        """Enhanced narrative generation with style control"""

        style_prompts = {
            "professional": "Write in a professional, business-like tone suitable for tax professionals.",
            "academic": "Write in an academic style suitable for research papers and scholarly work.",
            "conversational": "Write in a clear, conversational style that's accessible to general audiences.",
            "legal": "Write in formal legal language appropriate for court documents and legal briefs."
        }

        style_instruction = style_prompts.get(style, style_prompts["professional"])

        enhanced_prompt = f"""
        {style_instruction}

        Please enhance this tax law case narrative:
        {case.narrative}

        Requirements:
        - Maintain all factual accuracy
        - Use appropriate {style} terminology
        - Ensure logical flow and clarity
        - Keep all monetary amounts and legal details exact
        """

        return self.generate_text_with_fallback(enhanced_prompt, temperature=0.6)


def demonstrate_basic_ai_integration(api_key: str):
    """Demonstrate basic AI integration setup and usage"""

    logger.info("=" * 60)
    logger.info("BASIC AI INTEGRATION DEMONSTRATION")
    logger.info("=" * 60)

    # Setup basic AI integration
    ai_config = AIConfig(
        provider="openai",
        model="gpt-4",
        api_key=api_key,
        temperature=0.4,
        max_tokens=1000
    )

    ai_integration = GenerativeAIIntegration(ai_config)

    # Generate a test case
    base_generator = TaxLawCaseGenerator()
    enhanced_generator = EnhancedTaxLawCaseGenerator(base_generator, ai_integration)

    logger.info("Generating test case with AI enhancement...")
    case = enhanced_generator.generate(
        complexity_level=ComplexityLevel.INTERMEDIATE,
        use_ai_enhancement=True
    )

    # Display results
    logger.info(f"✓ Generated case: {case.case_id}")
    logger.info(f"  - Title: {case.title}")
    logger.info(f"  - Complexity: {case.complexity_level.value}")
    logger.info(f"  - Entities: {len(case.entities)}")
    logger.info(f"  - Events: {len(case.events)}")
    logger.info(f"  - Reasoning steps: {len(case.reasoning_chain)}")
    logger.info(f"  - Narrative length: {len(case.narrative.split())} words")

    # Show narrative sample
    narrative_preview = case.narrative[:200] + "..." if len(case.narrative) > 200 else case.narrative
    logger.info(f"  - Narrative preview: {narrative_preview}")

    return case


def demonstrate_multi_provider_integration(primary_api_key: str, fallback_api_key: Optional[str] = None):
    """Demonstrate multi-provider AI integration with fallback"""

    logger.info("=" * 60)
    logger.info("MULTI-PROVIDER AI INTEGRATION DEMONSTRATION")
    logger.info("=" * 60)

    # Setup primary provider (OpenAI)
    primary_config = AIConfig(
        provider="openai",
        model="gpt-4",
        api_key=primary_api_key,
        temperature=0.3
    )

    # Setup fallback provider (could be another OpenAI instance or custom provider)
    fallback_config = None
    if fallback_api_key:
        fallback_config = AIConfig(
            provider="openai",
            model="gpt-3.5-turbo",  # Cheaper fallback model
            api_key=fallback_api_key,
            temperature=0.4
        )
    else:
        # Use custom provider as fallback
        fallback_config = AIConfig(
            provider="custom",
            model="custom-model",
            api_key="not-required"
        )

    # Initialize multi-provider integration
    multi_ai = MultiProviderAIIntegration(primary_config, fallback_config)

    # Generate case with enhanced integration
    base_generator = TaxLawCaseGenerator()
    case = base_generator.generate(complexity_level=ComplexityLevel.ADVANCED)

    logger.info("Testing multi-provider narrative enhancement...")

    # Test different styles
    styles = ["professional", "academic", "conversational"]
    enhanced_narratives = {}

    for style in styles:
        logger.info(f"Generating {style} style narrative...")
        try:
            enhanced_narrative = multi_ai.enhance_narrative_with_style(case, style)
            enhanced_narratives[style] = enhanced_narrative
            logger.info(f"✓ {style} style completed ({len(enhanced_narrative.split())} words)")
        except Exception as e:
            logger.error(f"✗ {style} style failed: {e}")

    return case, enhanced_narratives


def demonstrate_custom_validation_workflow(api_key: str):
    """Demonstrate custom validation and quality control workflow"""

    logger.info("=" * 60)
    logger.info("CUSTOM VALIDATION WORKFLOW DEMONSTRATION")
    logger.info("=" * 60)

    # Setup AI integration with custom validation
    ai_config = AIConfig(
        provider="openai",
        model="gpt-4",
        api_key=api_key,
        reasoning_validation_temperature=0.1,  # Very precise for validation
        max_tokens=1500
    )

    ai_integration = GenerativeAIIntegration(ai_config)

    # Generate multiple cases and validate them
    base_generator = TaxLawCaseGenerator()
    validation_results = []

    for i in range(3):  # Generate 3 test cases
        logger.info(f"Generating and validating case {i + 1}/3...")

        # Generate base case
        case = base_generator.generate(complexity_level=ComplexityLevel.EXPERT)

        # Enhance with AI
        case.narrative = ai_integration.enhance_narrative(case)
        case.reasoning_chain = ai_integration.generate_advanced_reasoning(case)

        # Validate with AI
        validation = ai_integration.validate_reasoning_chain(case)
        complexity_assessment = ai_integration.assess_case_complexity(case)

        result = {
            'case_id': case.case_id,
            'validation_score': validation.get('overall_confidence_score', 0),
            'complexity_score': complexity_assessment.get('overall_complexity', 0),
            'is_logically_sound': validation.get('is_logically_sound', False),
            'narrative_length': len(case.narrative.split()),
            'reasoning_steps': len(case.reasoning_chain)
        }

        validation_results.append(result)

        logger.info(f"  Case {case.case_id[:8]}: Validation={result['validation_score']:.2f}, "
                    f"Complexity={result['complexity_score']:.2f}, "
                    f"Sound={result['is_logically_sound']}")

    # Summary statistics
    avg_validation = sum(r['validation_score'] for r in validation_results) / len(validation_results)
    avg_complexity = sum(r['complexity_score'] for r in validation_results) / len(validation_results)
    sound_cases = sum(1 for r in validation_results if r['is_logically_sound'])

    logger.info(f"Validation Summary:")
    logger.info(f"  - Average validation score: {avg_validation:.3f}")
    logger.info(f"  - Average complexity score: {avg_complexity:.3f}")
    logger.info(f"  - Logically sound cases: {sound_cases}/{len(validation_results)}")

    return validation_results


def save_demonstration_results(results: Dict[str, Any], output_file: str = "custom_integration_results.json"):
    """Save demonstration results to file"""

    output_path = Path(output_file)

    # Prepare results for JSON serialization
    serializable_results = {}

    for key, value in results.items():
        if hasattr(value, '__dict__'):
            # Convert objects to dictionaries
            serializable_results[key] = value.__dict__
        elif isinstance(value, list) and value and hasattr(value[0], '__dict__'):
            # Convert list of objects to list of dictionaries
            serializable_results[key] = [item.__dict__ for item in value]
        else:
            serializable_results[key] = value

    # Add metadata
    serializable_results['metadata'] = {
        'generation_timestamp': datetime.now().isoformat(),
        'demonstration_type': 'custom_ai_integration',
        'total_cases_generated': len([k for k, v in results.items() if 'case' in k.lower()])
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


def main():
    """Main demonstration function"""

    import argparse

    parser = argparse.ArgumentParser(description="Custom AI Integration Demonstrations")
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--fallback-key', help='Fallback API key for multi-provider demo')
    parser.add_argument('--demo', choices=['basic', 'multi', 'validation', 'all'],
                        default='all', help='Which demonstration to run')
    parser.add_argument('--output', default='custom_integration_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key required. Use --api-key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    results = {}

    try:
        logger.info("Starting Custom AI Integration Demonstrations")
        logger.info(f"Timestamp: {datetime.now()}")

        if args.demo in ['basic', 'all']:
            logger.info("\n Running Basic AI Integration Demo...")
            basic_case = demonstrate_basic_ai_integration(api_key)
            results['basic_integration'] = basic_case

        if args.demo in ['multi', 'all']:
            logger.info("\n Running Multi-Provider Integration Demo...")
            multi_case, multi_narratives = demonstrate_multi_provider_integration(
                api_key, args.fallback_key
            )
            results['multi_provider'] = {
                'case': multi_case,
                'enhanced_narratives': multi_narratives
            }

        if args.demo in ['validation', 'all']:
            logger.info("\n Running Custom Validation Workflow Demo...")
            validation_results = demonstrate_custom_validation_workflow(api_key)
            results['validation_workflow'] = validation_results

        # Save results
        save_demonstration_results(results, args.output)

        logger.info("=" * 60)
        logger.info("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {args.output}")
        logger.info("Check the output file for detailed results and generated cases.")

    except KeyboardInterrupt:
        logger.warning("\n Demonstrations interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
