"""
Configuration System and Evaluation Framework
Extends the Tax Law Reasoning Data Generator with configuration management and evaluation capabilities
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import logging

# Import from main module (assume tax_law_generator.py exists)
from .tax_law_generator import TaxLawCase, ComplexityLevel, TaxLawCaseGenerator

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for case generation"""

    # Generation parameters
    complexity_distribution: Dict[str, float] = None  # Distribution of complexity levels
    min_entities_per_case: int = 1
    max_entities_per_case: int = 4
    min_events_per_case: int = 2
    max_events_per_case: int = 8

    # Domain-specific parameters
    tax_year: str = "2024"
    jurisdiction: str = "US_Federal"
    applicable_tax_codes: List[str] = None

    # Narrative generation parameters
    narrative_length_target: int = 500  # Target words
    include_distractors: bool = True  # Include irrelevant information
    formality_level: str = "professional"  # casual, professional, legal

    # Reasoning chain parameters
    min_reasoning_steps: int = 3
    max_reasoning_steps: int = 8
    include_intermediate_calculations: bool = True
    show_confidence_scores: bool = False

    # Output parameters
    output_format: str = "json"  # json, yaml, csv
    include_metadata: bool = True

    def __post_init__(self):
        # Set defaults
        if self.complexity_distribution is None:
            self.complexity_distribution = {
                "basic": 0.3,
                "intermediate": 0.4,
                "advanced": 0.2,
                "expert": 0.1
            }

        if self.applicable_tax_codes is None:
            self.applicable_tax_codes = [
                "IRC_61",  # Gross Income
                "IRC_162",  # Business Deductions
                "IRC_163",  # Interest Deduction
                "IRC_164",  # State and Local Taxes
                "IRC_165",  # Losses
                "IRC_170",  # Charitable Contributions
            ]


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = GenerationConfig()

    def load_config(self, config_path: str) -> GenerationConfig:
        """Load configuration from file"""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")

        # Convert dict to GenerationConfig
        self.config = GenerationConfig(**config_dict)
        return self.config

    def save_config(self, config: GenerationConfig, save_path: str):
        """Save configuration to file"""
        path = Path(save_path)
        config_dict = asdict(config)

        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")

        logger.info(f"Configuration saved to {save_path}")


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

    # Metadata
    generation_time: float = 0.0
    evaluation_timestamp: str = ""


class CaseEvaluator:
    """Evaluates generated tax law cases for quality and difficulty"""

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.evaluation_criteria = self._initialize_evaluation_criteria()

    def evaluate_case(self, case: 'TaxLawCase') -> EvaluationMetrics:
        """Comprehensive evaluation of a generated case"""

        metrics = EvaluationMetrics(
            case_id=case.case_id,
            complexity_level=case.complexity_level.value,
            num_entities=len(case.entities),
            num_events=len(case.events),
            num_reasoning_steps=len(case.reasoning_chain),
            narrative_length=len(case.narrative.split()),
            evaluation_timestamp=datetime.now().isoformat()
        )

        # Evaluate narrative coherence
        metrics.narrative_coherence_score = self._evaluate_narrative_coherence(case)

        # Evaluate reasoning validity
        metrics.reasoning_validity_score = self._evaluate_reasoning_validity(case)

        # Evaluate tax law accuracy
        metrics.tax_law_accuracy_score = self._evaluate_tax_accuracy(case)

        # Calculate overall quality score
        metrics.overall_quality_score = self._calculate_overall_quality(metrics)

        # Estimate difficulty
        metrics.estimated_difficulty = self._estimate_difficulty(case)

        # Estimate human solvability
        metrics.human_solvability_score = self._estimate_human_solvability(case)

        return metrics

    def _evaluate_narrative_coherence(self, case: 'TaxLawCase') -> float:
        """Evaluate how coherent and well-structured the narrative is"""
        score = 0.0

        # Check if narrative mentions all entities
        entity_mentions = sum(1 for entity in case.entities if entity.name in case.narrative)
        score += (entity_mentions / len(case.entities)) * 0.3

        # Check if narrative describes all events
        event_mentions = sum(1 for event in case.events if event.event_type.replace('_', ' ') in case.narrative.lower())
        score += (event_mentions / len(case.events)) * 0.4

        # Check narrative length appropriateness
        word_count = len(case.narrative.split())
        target_length = self.config.narrative_length_target
        length_ratio = min(word_count / target_length, target_length / word_count)
        score += length_ratio * 0.3

        return min(score, 1.0)

    def _evaluate_reasoning_validity(self, case: 'TaxLawCase') -> float:
        """Evaluate the validity of the reasoning chain"""
        if not case.reasoning_chain:
            return 0.0

        score = 0.0

        # Check if reasoning chain is complete
        step_types = {step.step_type for step in case.reasoning_chain}
        required_steps = {"fact_identification", "rule_application", "calculation", "conclusion"}
        completeness = len(step_types.intersection(required_steps)) / len(required_steps)
        score += completeness * 0.5

        # Check logical flow
        for i, step in enumerate(case.reasoning_chain):
            if step.reasoning_text and len(step.reasoning_text) > 10:
                score += 0.1  # Each step with meaningful reasoning adds points

        # Normalize score
        score = min(score, 1.0)

        return score

    def _evaluate_tax_accuracy(self, case: 'TaxLawCase') -> float:
        """Evaluate the accuracy of tax law application"""
        score = 0.0

        # Check if applicable tax codes are mentioned in reasoning
        mentioned_codes = 0
        for step in case.reasoning_chain:
            if "IRC" in step.reasoning_text:
                mentioned_codes += 1

        if mentioned_codes > 0:
            score += 0.4

        # Check calculation accuracy (simplified check)
        total_income = sum(e.amount for e in case.events if "income" in e.event_type)
        total_deductions = sum(e.amount for e in case.events if "deduction" in e.event_type)

        if total_income > 0:
            score += 0.3  # Case has income to work with

        if total_deductions > 0:
            score += 0.2  # Case has deductions to consider

        # Check if ground truth seems reasonable
        if case.ground_truth_answer and "$" in case.ground_truth_answer:
            score += 0.1

        return min(score, 1.0)

    def _calculate_overall_quality(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall quality score"""
        weights = {
            'narrative_coherence': 0.3,
            'reasoning_validity': 0.4,
            'tax_accuracy': 0.3
        }

        score = (
                metrics.narrative_coherence_score * weights['narrative_coherence'] +
                metrics.reasoning_validity_score * weights['reasoning_validity'] +
                metrics.tax_law_accuracy_score * weights['tax_accuracy']
        )

        return score

    def _estimate_difficulty(self, case: 'TaxLawCase') -> float:
        """Estimate the difficulty of the case"""
        difficulty = 0.0

        # Base difficulty from complexity level
        complexity_weights = {
            'basic': 0.2,
            'intermediate': 0.4,
            'advanced': 0.6,
            'expert': 0.8
        }
        difficulty += complexity_weights.get(case.complexity_level.value, 0.5)

        # Adjust for number of entities and events
        difficulty += (len(case.entities) - 1) * 0.05
        difficulty += (len(case.events) - 2) * 0.03

        # Adjust for reasoning chain complexity
        difficulty += (len(case.reasoning_chain) - 3) * 0.02

        return min(difficulty, 1.0)

    def _estimate_human_solvability(self, case: 'TaxLawCase') -> float:
        """Estimate how likely a human expert is to solve this correctly"""
        base_solvability = 1.0

        # Reduce solvability based on complexity
        complexity_penalties = {
            'basic': 0.0,
            'intermediate': 0.1,
            'advanced': 0.3,
            'expert': 0.5
        }
        base_solvability -= complexity_penalties.get(case.complexity_level.value, 0.2)

        # Reduce based on number of interacting elements
        interaction_penalty = (len(case.entities) * len(case.events)) * 0.01
        base_solvability -= interaction_penalty

        return max(base_solvability, 0.1)  # Minimum 10% solvability

    def _initialize_evaluation_criteria(self) -> Dict[str, Any]:
        """Initialize evaluation criteria and thresholds"""
        return {
            'quality_thresholds': {
                'excellent': 0.9,
                'good': 0.7,
                'acceptable': 0.5,
                'poor': 0.3
            },
            'difficulty_thresholds': {
                'very_easy': 0.2,
                'easy': 0.4,
                'medium': 0.6,
                'hard': 0.8,
                'very_hard': 1.0
            }
        }


class DatasetGenerator:
    """Generates complete datasets of tax law cases"""

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.case_generator = None  # Will be initialized with TaxLawCaseGenerator
        self.evaluator = CaseEvaluator(config)
        self.generated_cases: List['TaxLawCase'] = []
        self.evaluation_results: List[EvaluationMetrics] = []

    def generate_dataset(self,
                         num_cases: int,
                         complexity_distribution: Dict[str, float] = None,
                         output_dir: str = "generated_dataset") -> Dict[str, Any]:
        """Generate a complete dataset of tax law cases"""

        # Use provided distribution or config default
        if complexity_distribution is None:
            complexity_distribution = self.config.complexity_distribution

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        dataset_info = {
            'generation_timestamp': datetime.now().isoformat(),
            'num_cases': num_cases,
            'complexity_distribution': complexity_distribution,
            'config': asdict(self.config),
            'cases': [],
            'evaluation_summary': {}
        }

        # Generate cases according to complexity distribution
        complexity_counts = self._calculate_complexity_counts(num_cases, complexity_distribution)

        logger.info(f"Generating {num_cases} cases with distribution: {complexity_counts}")

        for complexity_level, count in complexity_counts.items():
            for i in range(count):
                try:
                    # Generate case
                    case = self._generate_single_case(complexity_level)
                    self.generated_cases.append(case)

                    # Evaluate case
                    evaluation = self.evaluator.evaluate_case(case)
                    self.evaluation_results.append(evaluation)

                    # Add to dataset info
                    case_info = {
                        'case_id': case.case_id,
                        'complexity': complexity_level,
                        'evaluation': asdict(evaluation)
                    }
                    dataset_info['cases'].append(case_info)

                    logger.info(f"Generated case {i + 1}/{count} for {complexity_level}")

                except Exception as e:
                    logger.error(f"Error generating case: {e}")
                    continue

        # Generate evaluation summary
        dataset_info['evaluation_summary'] = self._generate_evaluation_summary()

        # Save dataset
        self._save_dataset(dataset_info, output_path)

        logger.info(f"Dataset generation complete. {len(self.generated_cases)} cases generated.")
        return dataset_info

    def _calculate_complexity_counts(self, total_cases: int, distribution: Dict[str, float]) -> Dict[str, int]:
        """Calculate how many cases to generate for each complexity level"""
        counts = {}
        remaining_cases = total_cases

        # Calculate counts for each complexity level
        for complexity, ratio in distribution.items():
            count = int(total_cases * ratio)
            counts[complexity] = count
            remaining_cases -= count

        # Distribute remaining cases
        complexity_levels = list(distribution.keys())
        for i in range(remaining_cases):
            complexity = complexity_levels[i % len(complexity_levels)]
            counts[complexity] += 1

        return counts

    def _generate_single_case(self, complexity_level: str) -> 'TaxLawCase':
        """Generate a single case of specified complexity"""
        # This would use the TaxLawCaseGenerator from the main module
        # For now, we'll create a mock implementation

        from tax_law_generator import TaxLawCaseGenerator, ComplexityLevel

        if self.case_generator is None:
            self.case_generator = TaxLawCaseGenerator(asdict(self.config))

        # Map string to enum
        complexity_map = {
            'basic': ComplexityLevel.BASIC,
            'intermediate': ComplexityLevel.INTERMEDIATE,
            'advanced': ComplexityLevel.ADVANCED,
            'expert': ComplexityLevel.EXPERT
        }

        complexity_enum = complexity_map.get(complexity_level, ComplexityLevel.BASIC)
        return self.case_generator.generate(complexity_level=complexity_enum)

    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the evaluation results"""
        if not self.evaluation_results:
            return {}

        df = pd.DataFrame([asdict(eval_result) for eval_result in self.evaluation_results])

        summary = {
            'total_cases': len(self.evaluation_results),
            'average_quality_score': df['overall_quality_score'].mean(),
            'average_difficulty': df['estimated_difficulty'].mean(),
            'average_human_solvability': df['human_solvability_score'].mean(),
            'quality_distribution': df['overall_quality_score'].describe().to_dict(),
            'complexity_distribution': df['complexity_level'].value_counts().to_dict(),
            'narrative_length_stats': df['narrative_length'].describe().to_dict()
        }

        return summary

    def _save_dataset(self, dataset_info: Dict[str, Any], output_path: Path):
        """Save the generated dataset to files"""

        # Save main dataset info
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        # Save individual cases
        cases_dir = output_path / "cases"
        cases_dir.mkdir(exist_ok=True)

        for case in self.generated_cases:
            case_file = cases_dir / f"{case.case_id}.json"
            case_dict = asdict(case)

            # Convert enums to strings for JSON serialization
            case_dict['complexity_level'] = case.complexity_level.value
            for step in case_dict['reasoning_chain']:
                step['step_type'] = step['step_type'].value if hasattr(step['step_type'], 'value') else step[
                    'step_type']

            with open(case_file, 'w') as f:
                json.dump(case_dict, f, indent=2)

        # Save evaluation results
        eval_df = pd.DataFrame([asdict(eval_result) for eval_result in self.evaluation_results])
        eval_df.to_csv(output_path / "evaluation_results.csv", index=False)

        logger.info(f"Dataset saved to {output_path}")


class GenerativeAIIntegration:
    """Integration with various generative AI libraries"""

    def __init__(self, ai_provider: str = "openai", api_key: str = None):
        self.ai_provider = ai_provider
        self.api_key = api_key
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the AI client based on provider"""
        if self.ai_provider == "openai":
            try:
                import openai
                return openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI library not installed. Install with: pip install openai")
                return None
        elif self.ai_provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("Anthropic library not installed. Install with: pip install anthropic")
                return None
        else:
            logger.warning(f"Unsupported AI provider: {self.ai_provider}")
            return None

    def enhance_narrative(self, base_narrative: str, enhancement_prompt: str = None) -> str:
        """Use generative AI to enhance the narrative quality"""
        if not self.client:
            return base_narrative

        default_prompt = f"""
        Please enhance this tax law case narrative to make it more realistic and engaging while maintaining accuracy:

        {base_narrative}

        Guidelines:
        - Keep all factual information intact
        - Make the language more natural and professional
        - Add realistic details that don't change the tax implications
        - Ensure the narrative flows well and is easy to understand
        """

        prompt = enhancement_prompt or default_prompt

        try:
            if self.ai_provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return response.choices[0].message.content
            elif self.ai_provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"Error enhancing narrative: {e}")
            return base_narrative

        return base_narrative

    def validate_reasoning_chain(self, case: 'TaxLawCase') -> Dict[str, Any]:
        """Use AI to validate the reasoning chain"""
        if not self.client:
            return {"valid": True, "feedback": "AI validation unavailable"}

        reasoning_text = "\n".join([f"Step {i + 1}: {step.reasoning_text}"
                                    for i, step in enumerate(case.reasoning_chain)])

        prompt = f"""
        Please validate this tax law reasoning chain for accuracy and completeness:

        Case: {case.narrative}

        Reasoning Chain:
        {reasoning_text}

        Ground Truth: {case.ground_truth_answer}

        Please provide:
        1. Is the reasoning logically sound? (Yes/No)
        2. Are there any errors in tax law application?
        3. Are any important steps missing?
        4. Overall confidence score (0-1)

        Respond in JSON format.
        """

        try:
            if self.ai_provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                # Parse JSON response
                import json
                return json.loads(response.choices[0].message.content)
            elif self.ai_provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                import json
                return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Error validating reasoning: {e}")
            return {"valid": True, "feedback": f"Validation error: {e}"}


# Example configuration files
def create_sample_configs():
    """Create sample configuration files"""

    # Basic configuration
    basic_config = GenerationConfig(
        complexity_distribution={"basic": 0.5, "intermediate": 0.3, "advanced": 0.2},
        min_entities_per_case=1,
        max_entities_per_case=2,
        narrative_length_target=300,
        include_distractors=False
    )

    # Advanced configuration
    advanced_config = GenerationConfig(
        complexity_distribution={"intermediate": 0.3, "advanced": 0.4, "expert": 0.3},
        min_entities_per_case=2,
        max_entities_per_case=4,
        narrative_length_target=800,
        include_distractors=True,
        show_confidence_scores=True
    )

    # Save configurations
    config_manager = ConfigManager()
    config_manager.save_config(basic_config, "configs/basic_config.json")
    config_manager.save_config(advanced_config, "configs/advanced_config.json")

    print("Sample configuration files created in configs/ directory")


# Main execution example
def main():
    """Example of using the complete system"""

    # Create sample configurations
    create_sample_configs()

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/basic_config.json")

    # Generate dataset
    dataset_gen = DatasetGenerator(config)
    dataset_info = dataset_gen.generate_dataset(
        num_cases=10,
        output_dir="sample_dataset"
    )

    print(f"Generated dataset with {dataset_info['num_cases']} cases")
    print(f"Average quality score: {dataset_info['evaluation_summary']['average_quality_score']:.2f}")


if __name__ == "__main__":
    main()