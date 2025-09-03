#!/usr/bin/env python3
"""
AI-Enhanced Tax Law Reasoning Data Generator - Quick Start Example
Fully integrated with OpenAI API for realistic case generation and validation
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
import argparse
import sys

from src.tax_law_generator.ai_integration import *
from src.tax_law_generator.tax_law_generator import *
from src.tax_law_generator.config_evaluation import  *
# from src.utils.data_structures import SomeDataClass


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_start_demo(api_key: str):
    """
    Demonstrate AI-enhanced tax law case generation with OpenAI API
    """
    print("=== AI-Enhanced Tax Law Reasoning Data Generator - Quick Start Demo ===\n")

    # Initialize AI integration with OpenAI API
    ai_config = AIConfig(
        provider="openai",
        model="gpt-4",
        api_key=api_key,
        temperature=0.3,
        max_tokens=1500
    )

    try:
        ai_integration = GenerativeAIIntegration(ai_config)
        print("✓ OpenAI API connection established")
    except Exception as e:
        print(f"✗ Failed to initialize OpenAI API: {e}")
        print("Please check your API key and try again.")
        return

    # 1. Generate AI-enhanced case
    print("\n1. Generating AI-enhanced tax law case...")

    base_generator = TaxLawCaseGenerator()
    enhanced_generator = EnhancedTaxLawCaseGenerator(base_generator, ai_integration)

    case = enhanced_generator.generate(
        complexity_level=ComplexityLevel.INTERMEDIATE,
        use_ai_enhancement=True
    )

    print(f"✓ Generated Case ID: {case.case_id}")
    print(f"  Title: {case.title}")
    print(f"  Complexity: {case.complexity_level.value}")
    print(f"  Entities: {len(case.entities)}")
    print(f"  Events: {len(case.events)}")
    print(f"  Reasoning Steps: {len(case.reasoning_chain)}")

    # Show AI-enhanced narrative
    narrative_preview = case.narrative[:300] + "..." if len(case.narrative) > 300 else case.narrative
    print(f"  AI-Enhanced Narrative: {narrative_preview}")

    # Show AI-generated reasoning chain
    print("\n  AI-Generated Reasoning Chain:")
    for i, step in enumerate(case.reasoning_chain[:3], 1):
        print(f"    Step {i} ({step.step_type.value}): {step.description}")
        reasoning_preview = step.reasoning_text[:150] + "..." if len(step.reasoning_text) > 150 else step.reasoning_text
        print(f"           {reasoning_preview}")

    print(f"\n  Ground Truth: {case.ground_truth_answer}\n")

    # 2. AI-powered case validation
    print("2. AI-powered reasoning validation...")

    validation_result = ai_integration.validate_reasoning_chain(case)

    print(f"✓ AI Validation Results:")
    print(f"  Logically Sound: {validation_result.get('is_logically_sound', 'Unknown')}")
    print(f"  Confidence Score: {validation_result.get('overall_confidence_score', 0):.2f}")
    print(f"  Technical Accuracy: {validation_result.get('technical_accuracy_score', 0):.2f}")

    if validation_result.get('tax_application_errors'):
        print(f"  Tax Errors Found: {len(validation_result['tax_application_errors'])}")

    if validation_result.get('missing_steps'):
        print(f"  Missing Steps: {len(validation_result['missing_steps'])}")

    # 3. Evaluate case quality
    print("\n3. Evaluating AI-enhanced case quality...")
    evaluator = CaseEvaluator()
    metrics = evaluator.evaluate_case(case)

    print(f"✓ Quality Evaluation:")
    print(f"  Overall Quality Score: {metrics.overall_quality_score:.2f}")
    print(f"  Narrative Coherence: {metrics.narrative_coherence_score:.2f}")
    print(f"  Reasoning Validity: {metrics.reasoning_validity_score:.2f}")
    print(f"  Tax Law Accuracy: {metrics.tax_law_accuracy_score:.2f}")
    print(f"  Estimated Difficulty: {metrics.estimated_difficulty:.2f}")
    print(f"  Human Solvability: {metrics.human_solvability_score:.2f}\n")

    # 4. AI complexity assessment
    print("4. AI complexity assessment...")

    complexity_assessment = ai_integration.assess_case_complexity(case)

    print(f"✓ AI Complexity Assessment:")
    print(f"  Cognitive Load: {complexity_assessment.get('cognitive_load', 0):.2f}")
    print(f"  Legal Complexity: {complexity_assessment.get('legal_complexity', 0):.2f}")
    print(f"  Calculation Difficulty: {complexity_assessment.get('calculation_difficulty', 0):.2f}")
    print(f"  Overall AI Complexity: {complexity_assessment.get('overall_complexity', 0):.2f}")

    # 5. Save enhanced case
    print("\n5. Saving AI-enhanced case...")

    case_dict = {
        'case_id': case.case_id,
        'title': case.title,
        'complexity_level': case.complexity_level.value,
        'ai_enhanced': True,
        'narrative': case.narrative,
        'entities': [{'id': e.id, 'name': e.name, 'type': e.entity_type,
                      'attributes': e.attributes} for e in case.entities],
        'events': [{'id': e.id, 'type': e.event_type, 'amount': e.amount,
                    'description': e.description, 'tax_implications': e.tax_implications}
                   for e in case.events],
        'reasoning_chain': [{'step_id': r.step_id, 'type': r.step_type.value,
                             'description': r.description, 'reasoning': r.reasoning_text,
                             'confidence': getattr(r, 'confidence_score', None)}
                            for r in case.reasoning_chain],
        'ground_truth_answer': case.ground_truth_answer,
        'ai_validation': validation_result,
        'ai_complexity_assessment': complexity_assessment,
        'quality_metrics': {
            'overall_quality': metrics.overall_quality_score,
            'narrative_coherence': metrics.narrative_coherence_score,
            'reasoning_validity': metrics.reasoning_validity_score,
            'tax_accuracy': metrics.tax_law_accuracy_score,
            'difficulty': metrics.estimated_difficulty,
            'human_solvability': metrics.human_solvability_score
        }
    }

    output_path = Path("demo_dataset/ai_enhanced_case.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(case_dict, f, indent=2)

    print(f"✓ AI-enhanced case saved to: {output_path}")

    print("\n=== AI-Enhanced Demo Complete! ===")
    print("The case has been fully generated and validated using OpenAI API.")


def advanced_ai_example(api_key: str):
    """
    Demonstrate advanced AI integration with custom configurations
    """
    print("\n=== Advanced AI Integration Example ===\n")

    # Load advanced configuration
    config_path = Path("configs/advanced_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        advanced_config = GenerationConfig(**config_data)
        print(f"✓ Loaded advanced configuration from {config_path}")
    else:
        # Use your provided advanced config
        advanced_config = GenerationConfig(
            complexity_distribution={"intermediate": 0.3, "advanced": 0.4, "expert": 0.3},
            min_entities_per_case=2,
            max_entities_per_case=4,
            min_events_per_case=4,
            max_events_per_case=8,
            narrative_length_target=800,
            include_distractors=True,
            min_reasoning_steps=5,
            max_reasoning_steps=10,
            applicable_tax_codes=["IRC_61", "IRC_162", "IRC_163", "IRC_170", "IRC_199A"]
        )
        print("✓ Using built-in advanced configuration")

    # Initialize AI with advanced settings
    ai_config = AIConfig(
        provider="openai",
        model="gpt-4",
        api_key=api_key,
        temperature=0.3,
        narrative_enhancement_temperature=0.6,
        reasoning_validation_temperature=0.1,
        reasoning_generation_temperature=0.4,
        max_tokens=2000
    )

    ai_integration = GenerativeAIIntegration(ai_config)

    # Generate expert-level case
    print("1. Generating expert-level case with advanced AI processing...")

    base_generator = TaxLawCaseGenerator(config=advanced_config.__dict__)
    enhanced_generator = EnhancedTaxLawCaseGenerator(base_generator, ai_integration)

    expert_case = enhanced_generator.generate(
        complexity_level=ComplexityLevel.EXPERT,
        use_ai_enhancement=True
    )

    print(f"✓ Generated Expert Case: {expert_case.case_id}")
    print(f"  Entities: {len(expert_case.entities)}")
    print(f"  Events: {len(expert_case.events)}")
    print(f"  AI Reasoning Steps: {len(expert_case.reasoning_chain)}")
    print(f"  Narrative Word Count: {len(expert_case.narrative.split())}")

    # Comprehensive AI analysis
    print("\n2. Comprehensive AI analysis...")

    # Detailed validation
    validation = ai_integration.validate_reasoning_chain(expert_case)
    complexity_assessment = ai_integration.assess_case_complexity(expert_case)

    print(f"✓ Advanced AI Analysis:")
    print(f"  Validation Confidence: {validation.get('overall_confidence_score', 0):.3f}")
    print(f"  Technical Accuracy: {validation.get('technical_accuracy_score', 0):.3f}")
    print(f"  Completeness Score: {validation.get('completeness_score', 0):.3f}")
    print(f"  AI-Assessed Complexity: {complexity_assessment.get('overall_complexity', 0):.3f}")

    if validation.get('strengths'):
        print(f"  AI-Identified Strengths: {len(validation['strengths'])}")

    if validation.get('improvement_suggestions'):
        print(f"  AI Improvement Suggestions: {len(validation['improvement_suggestions'])}")

    # Export comprehensive results
    comprehensive_export = {
        'generation_metadata': {
            'api_model': ai_config.model,
            'enhancement_level': 'expert',
            'config_source': 'advanced_config.json'
        },
        'case_data': {
            'case_id': expert_case.case_id,
            'complexity_level': expert_case.complexity_level.value,
            'narrative': expert_case.narrative,
            'reasoning_chain': [
                {
                    'step_id': r.step_id,
                    'type': r.step_type.value,
                    'reasoning': r.reasoning_text,
                    'input_data': r.input_data,
                    'output_data': r.output_data
                } for r in expert_case.reasoning_chain
            ],
            'ground_truth': expert_case.ground_truth_answer
        },
        'ai_analysis': {
            'validation_results': validation,
            'complexity_assessment': complexity_assessment
        }
    }

    export_path = Path("demo_dataset/expert_case_comprehensive.json")
    with open(export_path, 'w') as f:
        json.dump(comprehensive_export, f, indent=2)

    print(f"✓ Comprehensive analysis exported to: {export_path}")
    print("\n=== Advanced AI Example Complete! ===")


def generate_ai_dataset(api_key: str, config_file: str = None, num_cases: int = 10, output_dir: str = "ai_dataset"):
    """
    Generate a complete dataset using AI enhancement
    """
    print(f"\n=== Generating AI-Enhanced Dataset ({num_cases} cases) ===\n")

    # Load configuration
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        config = GenerationConfig(**config_data)
        print(f"✓ Loaded configuration from {config_file}")
    else:
        # Use basic config as fallback
        config = GenerationConfig(
            complexity_distribution={"basic": 0.5, "intermediate": 0.3, "advanced": 0.2},
            narrative_length_target=600,
            include_distractors=True
        )
        print("✓ Using default configuration")

    # Initialize AI
    ai_config = AIConfig(provider="openai", model="gpt-4", api_key=api_key)
    ai_integration = GenerativeAIIntegration(ai_config)

    # Create enhanced dataset generator
    base_generator = TaxLawCaseGenerator(config=config.__dict__)
    enhanced_generator = EnhancedTaxLawCaseGenerator(base_generator, ai_integration)

    # Generate cases with AI enhancement
    print("Generating AI-enhanced cases...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    cases_data = []
    validation_results = []

    complexity_counts = {
        'basic': int(num_cases * config.complexity_distribution.get('basic', 0)),
        'intermediate': int(num_cases * config.complexity_distribution.get('intermediate', 0)),
        'advanced': int(num_cases * config.complexity_distribution.get('advanced', 0)),
        'expert': int(num_cases * config.complexity_distribution.get('expert', 0))
    }

    # Adjust for rounding
    total_assigned = sum(complexity_counts.values())
    if total_assigned < num_cases:
        complexity_counts['intermediate'] += (num_cases - total_assigned)

    complexity_map = {
        'basic': ComplexityLevel.BASIC,
        'intermediate': ComplexityLevel.INTERMEDIATE,
        'advanced': ComplexityLevel.ADVANCED,
        'expert': ComplexityLevel.EXPERT
    }

    case_counter = 0
    for complexity, count in complexity_counts.items():
        if count == 0:
            continue

        for i in range(count):
            try:
                case = enhanced_generator.generate(
                    complexity_level=complexity_map[complexity],
                    use_ai_enhancement=True
                )

                # AI validation
                validation = ai_integration.validate_reasoning_chain(case)
                complexity_assessment = ai_integration.assess_case_complexity(case)

                case_data = {
                    'case_id': case.case_id,
                    'complexity_level': case.complexity_level.value,
                    'narrative': case.narrative,
                    'entities_count': len(case.entities),
                    'events_count': len(case.events),
                    'reasoning_steps_count': len(case.reasoning_chain),
                    'ai_validation_score': validation.get('overall_confidence_score', 0),
                    'ai_complexity_score': complexity_assessment.get('overall_complexity', 0)
                }

                cases_data.append(case_data)
                validation_results.append(validation)

                case_counter += 1
                print(f"  Generated case {case_counter}/{num_cases} ({complexity})")

                # Save individual case
                individual_case_path = output_path / f"case_{case.case_id}.json"
                case_export = {
                    'case_data': case_data,
                    'full_case': {
                        'narrative': case.narrative,
                        'reasoning_chain': [r.reasoning_text for r in case.reasoning_chain],
                        'ground_truth': case.ground_truth_answer
                    },
                    'ai_validation': validation,
                    'ai_complexity': complexity_assessment
                }

                with open(individual_case_path, 'w') as f:
                    json.dump(case_export, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to generate case {i + 1} for {complexity}: {e}")
                continue

    # Generate dataset summary
    dataset_summary = {
        'generation_timestamp': pd.Timestamp.now().isoformat(),
        'total_cases': len(cases_data),
        'ai_model_used': ai_config.model,
        'configuration': config.__dict__,
        'cases_overview': cases_data,
        'quality_metrics': {
            'average_validation_score': sum(c['ai_validation_score'] for c in cases_data) / len(
                cases_data) if cases_data else 0,
            'average_complexity_score': sum(c['ai_complexity_score'] for c in cases_data) / len(
                cases_data) if cases_data else 0,
            'complexity_distribution': {k: sum(1 for c in cases_data if c['complexity_level'] == k) for k in
                                        ['basic', 'intermediate', 'advanced', 'expert']}
        }
    }

    with open(output_path / "dataset_summary.json", 'w') as f:
        json.dump(dataset_summary, f, indent=2)

    print(f"\n✓ Generated {len(cases_data)} AI-enhanced cases")
    print(f"✓ Average AI validation score: {dataset_summary['quality_metrics']['average_validation_score']:.2f}")
    print(f"✓ Dataset saved to: {output_dir}/")

    return dataset_summary


def create_cli():
    """
    Create command-line interface for AI-enhanced generator
    """
    parser = argparse.ArgumentParser(description="AI-Enhanced Tax Law Reasoning Data Generator")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run AI-enhanced demonstration')
    demo_parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')

    # Generate single case command
    generate_parser = subparsers.add_parser('generate', help='Generate single AI-enhanced case')
    generate_parser.add_argument('--complexity', choices=['basic', 'intermediate', 'advanced', 'expert'],
                                 default='intermediate', help='Case complexity level')
    generate_parser.add_argument('--output', default='ai_generated_case.json', help='Output file path')
    generate_parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')

    # Generate dataset command
    dataset_parser = subparsers.add_parser('dataset', help='Generate AI-enhanced dataset')
    dataset_parser.add_argument('--num-cases', type=int, default=10, help='Number of cases to generate')
    dataset_parser.add_argument('--config', help='Configuration file path')
    dataset_parser.add_argument('--output-dir', default='ai_generated_dataset', help='Output directory')
    dataset_parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')

    return parser


def main():
    """
    Main entry point for CLI
    """
    parser = create_cli()
    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("Error: OpenAI API key required.")
        print("Provide it via --api-key argument or set OPENAI_API_KEY environment variable.")
        return

    if args.command == 'generate':
        complexity_map = {
            'basic': ComplexityLevel.BASIC,
            'intermediate': ComplexityLevel.INTERMEDIATE,
            'advanced': ComplexityLevel.ADVANCED,
            'expert': ComplexityLevel.EXPERT
        }

        ai_config = AIConfig(provider="openai", model="gpt-4", api_key=api_key)
        ai_integration = GenerativeAIIntegration(ai_config)

        base_generator = TaxLawCaseGenerator()
        enhanced_generator = EnhancedTaxLawCaseGenerator(base_generator, ai_integration)

        case = enhanced_generator.generate(
            complexity_level=complexity_map[args.complexity],
            use_ai_enhancement=True
        )

        # Save with full AI analysis
        validation = ai_integration.validate_reasoning_chain(case)
        complexity_assessment = ai_integration.assess_case_complexity(case)

        case_dict = {
            'case_id': case.case_id,
            'title': case.title,
            'complexity_level': case.complexity_level.value,
            'ai_enhanced': True,
            'narrative': case.narrative,
            'entities': [{'id': e.id, 'name': e.name, 'type': e.entity_type, 'attributes': e.attributes}
                         for e in case.entities],
            'events': [{'id': e.id, 'type': e.event_type, 'amount': e.amount, 'description': e.description}
                       for e in case.events],
            'reasoning_chain': [{'step_id': r.step_id, 'type': r.step_type.value, 'reasoning': r.reasoning_text}
                                for r in case.reasoning_chain],
            'ground_truth_answer': case.ground_truth_answer,
            'ai_validation': validation,
            'ai_complexity_assessment': complexity_assessment
        }

        with open(args.output, 'w') as f:
            json.dump(case_dict, f, indent=2)

        print(f"AI-enhanced case generated and saved to: {args.output}")
        print(f"AI validation score: {validation.get('overall_confidence_score', 0):.2f}")

    elif args.command == 'dataset':
        generate_ai_dataset(
            api_key=api_key,
            config_file=args.config,
            num_cases=args.num_cases,
            output_dir=args.output_dir
        )

    elif args.command == 'demo':
        quick_start_demo(api_key)
        advanced_ai_example(api_key)
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default demo mode - check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            quick_start_demo(api_key)
            print("\n" + "=" * 60)
            print("For more features:")
            print("Run: python quickstart_example.py demo --api-key YOUR_KEY")
            print("Or use CLI: python quickstart_example.py --help")
        else:
            print("OpenAI API key required for demo.")
            print("Set OPENAI_API_KEY environment variable or use:")
            print("python quickstart_example.py demo --api-key YOUR_KEY")
    else:
        main()
