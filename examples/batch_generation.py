"""
Batch Generation Example for Tax Law Reasoning Data Generator
Demonstrates large-scale dataset generation with comprehensive logging and evaluation
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax_law_generator.config_evaluation import (
    ConfigManager,
    GenerationConfig,
    DatasetGenerator
)
from src.tax_law_generator.ai_integration import (
    GenerativeAIIntegration,
    AIConfig,
    EnhancedTaxLawCaseGenerator
)
from src.tax_law_generator.tax_law_generator import TaxLawCaseGenerator


# Configure comprehensive logging
def setup_logging(output_dir: str):
    """Setup detailed logging for batch generation"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_generation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Batch generation started at {datetime.now()}")
    return logger


def load_batch_configuration(config_name: str = "advanced") -> GenerationConfig:
    """Load configuration for batch generation"""
    try:
        config_manager = ConfigManager()

        # Try to load from templates directory first
        try:
            config = config_manager.load_template_config(config_name)
            logging.info(f"Loaded {config_name} configuration from templates")
            return config
        except FileNotFoundError:
            # Fall back to default configuration
            logging.warning(f"Template {config_name} not found, using default configuration")
            return GenerationConfig()

    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        logging.info("Using default configuration")
        return GenerationConfig()


def run_batch_generation(
        num_cases: int = 50,
        output_dir: str = "batch_output",
        config_name: str = "advanced",
        use_ai_enhancement: bool = True,
        api_key: str = os.getenv('OPENAI_API_KEY')
):
    """
    Run comprehensive batch generation with AI enhancement

    Args:
        num_cases: Number of cases to generate
        output_dir: Output directory for results
        config_name: Configuration template to use
        use_ai_enhancement: Whether to use AI enhancement
        api_key: OpenAI API key for enhancement
    """

    logger = setup_logging(output_dir)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_batch_configuration(config_name)

        logger.info(f"Configuration loaded:")
        logger.info(f"  - Complexity distribution: {config.complexity_distribution}")
        logger.info(f"  - Entities per case: {config.min_entities_per_case}-{config.max_entities_per_case}")
        logger.info(f"  - Events per case: {config.min_events_per_case}-{config.max_events_per_case}")
        logger.info(f"  - Narrative target length: {config.narrative_length_target}")

        # Setup generators
        if use_ai_enhancement and api_key:
            logger.info("Initializing AI-enhanced generation...")

            ai_config = AIConfig(
                provider="openai",
                model="gpt-4",
                api_key=api_key,
                temperature=0.3,
                max_tokens=1500
            )

            ai_integration = GenerativeAIIntegration(ai_config)
            base_generator = TaxLawCaseGenerator(config.__dict__)
            enhanced_generator = EnhancedTaxLawCaseGenerator(base_generator, ai_integration)

            # Test AI connection
            test_case = base_generator.generate()
            test_narrative = ai_integration.enhance_narrative(test_case)
            if test_narrative != test_case.narrative:
                logger.info("✓ AI enhancement connection successful")
            else:
                logger.warning("AI enhancement may not be working properly")

        else:
            logger.info("Using base generation without AI enhancement")
            enhanced_generator = None

        # Initialize dataset generator
        dataset_gen = DatasetGenerator(config)

        # Override case generator if using AI enhancement
        if enhanced_generator:
            # Monkey patch to use enhanced generator
            original_generate = dataset_gen._generate_single_case

            def enhanced_generate_single_case(complexity_level: str):
                from src.tax_law_generator.tax_law_generator import ComplexityLevel
                complexity_map = {
                    'basic': ComplexityLevel.BASIC,
                    'intermediate': ComplexityLevel.INTERMEDIATE,
                    'advanced': ComplexityLevel.ADVANCED,
                    'expert': ComplexityLevel.EXPERT
                }
                return enhanced_generator.generate(
                    complexity_level=complexity_map.get(complexity_level, ComplexityLevel.BASIC),
                    use_ai_enhancement=True
                )

            dataset_gen._generate_single_case = enhanced_generate_single_case

        # Generate dataset
        logger.info(f"Starting batch generation of {num_cases} cases...")
        start_time = datetime.now()

        dataset_info = dataset_gen.generate_dataset(
            num_cases=num_cases,
            output_dir=output_dir
        )

        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()

        # Log results
        logger.info("=" * 60)
        logger.info("BATCH GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total cases generated: {dataset_info['num_cases']}")
        logger.info(f"Generation time: {generation_time:.2f} seconds")
        logger.info(f"Average time per case: {generation_time / num_cases:.2f} seconds")

        # Quality statistics
        eval_summary = dataset_info['evaluation_summary']
        logger.info(f"Quality Metrics:")
        logger.info(f"  - Average quality score: {eval_summary.get('average_quality_score', 0):.3f}")
        logger.info(f"  - Average difficulty: {eval_summary.get('average_difficulty', 0):.3f}")
        logger.info(f"  - Average human solvability: {eval_summary.get('average_human_solvability', 0):.3f}")

        # Complexity distribution
        complexity_dist = eval_summary.get('complexity_distribution', {})
        logger.info(f"Generated complexity distribution:")
        for complexity, count in complexity_dist.items():
            percentage = (count / dataset_info['num_cases']) * 100
            logger.info(f"  - {complexity}: {count} cases ({percentage:.1f}%)")

        # File locations
        logger.info(f"Output files:")
        logger.info(f"  - Dataset info: {output_dir}/dataset_info.json")
        logger.info(f"  - Individual cases: {output_dir}/cases/")
        logger.info(f"  - Evaluation results: {output_dir}/evaluation_results.csv")
        logger.info(f"  - Generation logs: {output_dir}/logs/")

        return dataset_info

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise


def analyze_batch_results(output_dir: str):
    """Analyze and report on batch generation results"""

    logger = logging.getLogger(__name__)

    # Load dataset info
    dataset_info_path = Path(output_dir) / "dataset_info.json"
    if not dataset_info_path.exists():
        logger.error("Dataset info file not found")
        return

    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)

    # Load evaluation results if available
    eval_path = Path(output_dir) / "evaluation_results.csv"
    if eval_path.exists():
        import pandas as pd
        eval_df = pd.DataFrame()
        try:
            eval_df = pd.read_csv(eval_path)
            logger.info(f"Loaded evaluation data for {len(eval_df)} cases")
        except Exception as e:
            logger.warning(f"Could not load evaluation CSV: {e}")

    # Detailed analysis
    logger.info("=" * 60)
    logger.info("DETAILED BATCH ANALYSIS")
    logger.info("=" * 60)

    eval_summary = dataset_info.get('evaluation_summary', {})

    if eval_summary:
        # Quality score distribution
        quality_dist = eval_summary.get('quality_distribution', {})
        logger.info("Quality Score Statistics:")
        for stat, value in quality_dist.items():
            logger.info(f"  - {stat}: {value:.3f}")

        # Narrative length statistics
        length_stats = eval_summary.get('narrative_length_stats', {})
        logger.info("Narrative Length Statistics:")
        for stat, value in length_stats.items():
            logger.info(f"  - {stat}: {value:.1f} words")

    # Count cases by directory
    cases_dir = Path(output_dir) / "cases"
    if cases_dir.exists():
        case_files = list(cases_dir.glob("*.json"))
        logger.info(f"Individual case files generated: {len(case_files)}")

        # Sample a few cases for spot checking
        if case_files:
            sample_case = case_files[0]
            with open(sample_case, 'r') as f:
                case_data = json.load(f)

            logger.info(f"Sample case ({sample_case.name}):")
            logger.info(f"  - Entities: {len(case_data.get('entities', []))}")
            logger.info(f"  - Events: {len(case_data.get('events', []))}")
            logger.info(f"  - Reasoning steps: {len(case_data.get('reasoning_chain', []))}")
            logger.info(f"  - Narrative length: {len(case_data.get('narrative', '').split())} words")


def main():
    """Main execution function with command line argument support"""

    import argparse

    parser = argparse.ArgumentParser(description="Batch Tax Law Case Generation")
    parser.add_argument('--cases', type=int, default=50, help='Number of cases to generate')
    parser.add_argument('--output', default='batch_output', help='Output directory')
    parser.add_argument('--config', default='advanced', help='Configuration template to use')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI enhancement')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze existing results')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    use_ai = not args.no_ai and api_key is not None

    if not use_ai:
        print("Running without AI enhancement (no API key provided)")

    try:
        if args.analyze_only:
            analyze_batch_results(args.output)
        else:
            # Run batch generation
            dataset_info = run_batch_generation(
                num_cases=args.cases,
                output_dir=args.output,
                config_name=args.config,
                use_ai_enhancement=use_ai,
                api_key=api_key
            )

            # Analyze results
            analyze_batch_results(args.output)

            print(f"\ Batch generation complete!")
            print(f"Generated {dataset_info['num_cases']} cases in '{args.output}' directory")
            print(f"Average quality score: {dataset_info['evaluation_summary']['average_quality_score']:.3f}")

    except KeyboardInterrupt:
        print("\n Batch generation interrupted by user")
    except Exception as e:
        print(f" Batch generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
