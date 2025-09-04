"""
Enhanced Configuration System and Evaluation Framework
Loads templates from external configs/templates directory with optimized, non-redundant code
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


# Utility functions to eliminate code duplication
def _read_config_file(path: Path) -> Dict[str, Any]:
    """Unified loader for JSON/YAML configuration files"""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix.lower() == '.json':
            return json.load(f)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")


def _save_config_file(data: Dict[str, Any], path: Path):
    """Unified saver for JSON/YAML configuration files"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix.lower() == '.json':
            json.dump(data, f, indent=2)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")


def _get_default_config_values() -> Dict[str, Any]:
    """Centralized default configuration values"""
    return {
        "complexity_distribution": {
            "basic": 0.3,
            "intermediate": 0.4,
            "advanced": 0.2,
            "expert": 0.1
        },
        "applicable_tax_codes": [
            "IRC_61",   # Gross Income
            "IRC_162",  # Business Deductions
            "IRC_163",  # Interest Deduction
            "IRC_164",  # State and Local Taxes
            "IRC_165",  # Losses
            "IRC_170",  # Charitable Contributions
        ]
    }


@dataclass
class GenerationConfig:
    """Configuration for case generation with external template loading"""

    # Generation parameters
    complexity_distribution: Dict[str, float] = None
    min_entities_per_case: int = 1
    max_entities_per_case: int = 4
    min_events_per_case: int = 2
    max_events_per_case: int = 8

    # Domain-specific parameters
    tax_year: str = "2024"
    jurisdiction: str = "US_Federal"
    applicable_tax_codes: List[str] = None

    # Narrative generation parameters
    narrative_length_target: int = 500
    include_distractors: bool = True
    formality_level: str = "professional"

    # Reasoning chain parameters
    min_reasoning_steps: int = 3
    max_reasoning_steps: int = 8
    include_intermediate_calculations: bool = True
    show_confidence_scores: bool = False

    # Output parameters
    output_format: str = "json"
    include_metadata: bool = True

    def __post_init__(self):
        """Set defaults using centralized default values"""
        defaults = _get_default_config_values()

        if self.complexity_distribution is None:
            self.complexity_distribution = defaults["complexity_distribution"]

        if self.applicable_tax_codes is None:
            self.applicable_tax_codes = defaults["applicable_tax_codes"]


class ConfigManager:
    """Enhanced configuration manager with template directory support"""

    def __init__(self, config_path: Optional[str] = None, templates_dir: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent.parent / 'configs' / 'templates'
        self.config = GenerationConfig()

    def load_config(self, config_path: Optional[str] = None) -> GenerationConfig:
        """Load configuration from file with template directory fallback"""
        # Determine config file path
        if config_path:
            path = Path(config_path)
        elif self.config_path:
            path = self.config_path
        else:
            # Try templates directory first
            path = self.templates_dir / "basic_config.json"
            if not path.exists():
                path = self.templates_dir / "advanced_config.json"

        if not path.exists():
            logger.warning(f"Config file {path} not found, using defaults")
            self.config = GenerationConfig()
            return self.config

        try:
            config_dict = _read_config_file(path)
            self.config = GenerationConfig(**config_dict)
            logger.info(f"Loaded configuration from {path}")
            return self.config
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            self.config = GenerationConfig()
            return self.config

    def save_config(self, config: GenerationConfig, save_path: Optional[str] = None):
        """Save configuration to file with smart path resolution"""
        if save_path:
            path = Path(save_path)
        else:
            # Default to templates directory
            path = self.templates_dir / "generated_config.json"

        try:
            _save_config_file(asdict(config), path)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving config to {path}: {e}")
            raise

    def load_template_config(self, template_name: str) -> GenerationConfig:
        """Load a specific template configuration by name"""
        template_path = self.templates_dir / f"{template_name}_config.json"

        if not template_path.exists():
            # Try YAML extension
            template_path = self.templates_dir / f"{template_name}_config.yaml"

        if not template_path.exists():
            raise FileNotFoundError(f"Template configuration '{template_name}' not found in {self.templates_dir}")

        return self.load_config(str(template_path))

    def list_available_templates(self) -> List[str]:
        """List available configuration templates"""
        if not self.templates_dir.exists():
            return []

        templates = []
        for file_path in self.templates_dir.glob("*_config.*"):
            template_name = file_path.stem.replace("_config", "")
            templates.append(template_name)

        return sorted(set(templates))


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
        self.evaluation_criteria = self._get_evaluation_criteria()

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

        # Apply evaluation methods
        metrics.narrative_coherence_score = self._evaluate_narrative_coherence(case)
        metrics.reasoning_validity_score = self._evaluate_reasoning_validity(case)
        metrics.tax_law_accuracy_score = self._evaluate_tax_accuracy(case)
        metrics.overall_quality_score = self._calculate_overall_quality(metrics)
        metrics.estimated_difficulty = self._estimate_difficulty(case)
        metrics.human_solvability_score = self._estimate_human_solvability(case)

        return metrics

    def _evaluate_narrative_coherence(self, case: 'TaxLawCase') -> float:
        """Evaluate narrative coherence and completeness"""
        score = 0.0

        # Entity mention coverage
        entity_mentions = sum(1 for entity in case.entities if entity.name in case.narrative)
        score += (entity_mentions / len(case.entities)) * 0.3

        # Event representation coverage
        event_mentions = sum(1 for event in case.events
                           if event.event_type.replace('_', ' ') in case.narrative.lower())
        score += (event_mentions / len(case.events)) * 0.4

        # Length appropriateness
        word_count = len(case.narrative.split())
        target_length = self.config.narrative_length_target
        length_ratio = min(word_count / target_length, target_length / word_count)
        score += length_ratio * 0.3

        return min(score, 1.0)

    def _evaluate_reasoning_validity(self, case: 'TaxLawCase') -> float:
        """Evaluate reasoning chain validity and completeness"""
        if not case.reasoning_chain:
            return 0.0

        score = 0.0

        # Required step coverage
        step_types = {step.step_type for step in case.reasoning_chain}
        required_steps = {"fact_identification", "rule_application", "calculation", "conclusion"}
        completeness = len(step_types.intersection(required_steps)) / len(required_steps)
        score += completeness * 0.5

        # Content quality check
        meaningful_steps = sum(1 for step in case.reasoning_chain
                             if step.reasoning_text and len(step.reasoning_text) > 10)
        score += min(meaningful_steps * 0.1, 0.5)

        return min(score, 1.0)

    def _evaluate_tax_accuracy(self, case: 'TaxLawCase') -> float:
        """Evaluate tax law accuracy and application"""
        score = 0.0

        # Tax code references
        irc_mentions = sum(1 for step in case.reasoning_chain if "IRC" in step.reasoning_text)
        score += min(irc_mentions * 0.1, 0.4)

        # Income/deduction presence
        total_income = sum(e.amount for e in case.events if "income" in e.event_type and e.amount)
        total_deductions = sum(e.amount for e in case.events if "deduction" in e.event_type and e.amount)

        if total_income > 0:
            score += 0.3
        if total_deductions > 0:
            score += 0.2

        # Ground truth format check
        if case.ground_truth_answer and "$" in case.ground_truth_answer:
            score += 0.1

        return min(score, 1.0)

    def _calculate_overall_quality(self, metrics: EvaluationMetrics) -> float:
        """Calculate weighted overall quality score"""
        weights = {'narrative_coherence': 0.3, 'reasoning_validity': 0.4, 'tax_accuracy': 0.3}

        return (
            metrics.narrative_coherence_score * weights['narrative_coherence'] +
            metrics.reasoning_validity_score * weights['reasoning_validity'] +
            metrics.tax_law_accuracy_score * weights['tax_accuracy']
        )

    def _estimate_difficulty(self, case: 'TaxLawCase') -> float:
        """Estimate case difficulty based on multiple factors"""
        complexity_weights = {'basic': 0.2, 'intermediate': 0.4, 'advanced': 0.6, 'expert': 0.8}

        difficulty = complexity_weights.get(case.complexity_level.value, 0.5)
        difficulty += (len(case.entities) - 1) * 0.05
        difficulty += (len(case.events) - 2) * 0.03
        difficulty += (len(case.reasoning_chain) - 3) * 0.02

        return min(difficulty, 1.0)

    def _estimate_human_solvability(self, case: 'TaxLawCase') -> float:
        """Estimate human expert solvability"""
        complexity_penalties = {'basic': 0.0, 'intermediate': 0.1, 'advanced': 0.3, 'expert': 0.5}

        base_solvability = 1.0 - complexity_penalties.get(case.complexity_level.value, 0.2)
        interaction_penalty = (len(case.entities) * len(case.events)) * 0.01

        return max(base_solvability - interaction_penalty, 0.1)

    def _get_evaluation_criteria(self) -> Dict[str, Any]:
        """Get evaluation criteria and thresholds"""
        return {
            'quality_thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3},
            'difficulty_thresholds': {'very_easy': 0.2, 'easy': 0.4, 'medium': 0.6, 'hard': 0.8, 'very_hard': 1.0}
        }


class DatasetGenerator:
    """Enhanced dataset generator with improved organization"""

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.case_generator = None
        self.evaluator = CaseEvaluator(config)
        self.generated_cases: List['TaxLawCase'] = []
        self.evaluation_results: List[EvaluationMetrics] = []

    def generate_dataset(self, num_cases: int, complexity_distribution: Dict[str, float] = None,
                        output_dir: str = "generated_dataset") -> Dict[str, Any]:
        """Generate complete dataset with evaluation"""

        complexity_distribution = complexity_distribution or self.config.complexity_distribution
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

        # Generate cases by complexity
        complexity_counts = self._calculate_complexity_counts(num_cases, complexity_distribution)
        logger.info(f"Generating {num_cases} cases with distribution: {complexity_counts}")

        for complexity_level, count in complexity_counts.items():
            for i in range(count):
                try:
                    case = self._generate_single_case(complexity_level)
                    evaluation = self.evaluator.evaluate_case(case)

                    self.generated_cases.append(case)
                    self.evaluation_results.append(evaluation)

                    dataset_info['cases'].append({
                        'case_id': case.case_id,
                        'complexity': complexity_level,
                        'evaluation': asdict(evaluation)
                    })

                    logger.info(f"Generated case {i + 1}/{count} for {complexity_level}")

                except Exception as e:
                    logger.error(f"Error generating case: {e}")
                    continue

        dataset_info['evaluation_summary'] = self._generate_evaluation_summary()
        self._save_dataset(dataset_info, output_path)

        logger.info(f"Dataset generation complete. {len(self.generated_cases)} cases generated.")
        return dataset_info

    def _calculate_complexity_counts(self, total_cases: int, distribution: Dict[str, float]) -> Dict[str, int]:
        """Calculate case counts per complexity level"""
        counts = {}
        remaining_cases = total_cases

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
        """Generate single case using appropriate generator"""
        if self.case_generator is None:
            self.case_generator = TaxLawCaseGenerator(asdict(self.config))

        complexity_map = {
            'basic': ComplexityLevel.BASIC,
            'intermediate': ComplexityLevel.INTERMEDIATE,
            'advanced': ComplexityLevel.ADVANCED,
            'expert': ComplexityLevel.EXPERT
        }

        complexity_enum = complexity_map.get(complexity_level, ComplexityLevel.BASIC)
        return self.case_generator.generate(complexity_level=complexity_enum)

    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of evaluation results"""
        if not self.evaluation_results:
            return {}

        df = pd.DataFrame([asdict(eval_result) for eval_result in self.evaluation_results])

        return {
            'total_cases': len(self.evaluation_results),
            'average_quality_score': df['overall_quality_score'].mean(),
            'average_difficulty': df['estimated_difficulty'].mean(),
            'average_human_solvability': df['human_solvability_score'].mean(),
            'quality_distribution': df['overall_quality_score'].describe().to_dict(),
            'complexity_distribution': df['complexity_level'].value_counts().to_dict(),
            'narrative_length_stats': df['narrative_length'].describe().to_dict()
        }

    def _save_dataset(self, dataset_info: Dict[str, Any], output_path: Path):
        """Save complete dataset with cases and evaluation results"""

        # Save dataset summary
        _save_config_file(dataset_info, output_path / "dataset_info.json")

        # Save individual cases
        cases_dir = output_path / "cases"
        cases_dir.mkdir(exist_ok=True)

        for case in self.generated_cases:
            case_dict = asdict(case)
            case_dict['complexity_level'] = case.complexity_level.value

            # Convert enum values for JSON serialization
            for step in case_dict['reasoning_chain']:
                if hasattr(step.get('step_type'), 'value'):
                    step['step_type'] = step['step_type'].value

            _save_config_file(case_dict, cases_dir / f"{case.case_id}.json")

        # Save evaluation results as CSV
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
        """Initialize AI client based on provider"""
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
        """Use generative AI to enhance narrative quality"""
        if not self.client:
            return base_narrative

        prompt = enhancement_prompt or f"""
        Please enhance this tax law case narrative to make it more realistic and engaging while maintaining accuracy:

        {base_narrative}

        Guidelines:
        - Keep all factual information intact
        - Make the language more natural and professional
        - Add realistic details that don't change the tax implications
        - Ensure the narrative flows well and is easy to understand
        """

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
        """Use AI to validate reasoning chain"""
        if not self.client:
            return {"valid": True, "feedback": "AI validation unavailable"}

        reasoning_text = "\n".join([f"Step {i + 1}: {step.reasoning_text}"
                                    for i, step in enumerate(case.reasoning_chain)])

        prompt = f"""
        Please validate this tax law reasoning chain for accuracy and completeness:

        Case: {case.narrative}
        Reasoning Chain: {reasoning_text}
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
                return json.loads(response.choices[0].message.content)
            elif self.ai_provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Error validating reasoning: {e}")
            return {"valid": True, "feedback": f"Validation error: {e}"}


# Configuration utilities
def create_sample_configs(templates_dir: Optional[str] = None):
    """Create sample configuration files in templates directory"""

    templates_path = Path(templates_dir) if templates_dir else Path("configs/templates")
    templates_path.mkdir(parents=True, exist_ok=True)

    configs = {
        "basic": GenerationConfig(
            complexity_distribution={"basic": 0.5, "intermediate": 0.3, "advanced": 0.2},
            min_entities_per_case=1,
            max_entities_per_case=2,
            narrative_length_target=300,
            include_distractors=False
        ),
        "advanced": GenerationConfig(
            complexity_distribution={"intermediate": 0.3, "advanced": 0.4, "expert": 0.3},
            min_entities_per_case=2,
            max_entities_per_case=4,
            narrative_length_target=800,
            include_distractors=True,
            show_confidence_scores=True
        )
    }

    config_manager = ConfigManager(templates_dir=str(templates_path))

    for name, config in configs.items():
        config_manager.save_config(config, str(templates_path / f"{name}_config.json"))

    logger.info(f"Sample configuration files created in {templates_path}")


# Main execution example
def main():
    """Example of using the complete enhanced system"""

    # Create sample configurations in templates directory
    create_sample_configs()

    # Load configuration from templates
    config_manager = ConfigManager()
    available_templates = config_manager.list_available_templates()
    logger.info(f"Available templates: {available_templates}")

    # Load specific template
    config = config_manager.load_template_config("basic")

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
