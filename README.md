# Synthetic Tax Law Reasoning Data Generator

A Python-based system for generating synthetic tax law cases to evaluate GenAI reasoning capabilities, based on the MuSR (Multistep Soft Reasoning) framework.

## 🎯 Project Overview

This system generates complex tax law scenarios with complete reasoning chains to test the limits of chain-of-thought reasoning in large language models. It addresses the need for scalable, challenging benchmarks that can evolve with advancing AI capabilities.

### Key Features

- **Scalable Complexity**: Generate cases from basic to expert level
- **Realistic Scenarios**: Natural language narratives based on real-world tax situations  
- **Complete Reasoning Chains**: Step-by-step logical progression with ground truth
- **Flexible Architecture**: Easily extensible for new domains and evaluation methods
- **AI Integration**: Built-in support for enhancing content with generative AI
- **Comprehensive Evaluation**: Multi-metric assessment of generated cases

## 📁 Project Structure

```
tax_law_reasoning_generator/
├── src/
│   ├── tax_law_generator.py          # Core generation engine
│   ├── config_evaluation.py          # Configuration & evaluation framework
│   ├── ai_integration.py             # GenAI library integrations
│   └── utils/
│       ├── data_structures.py        # Core data models
│       ├── validators.py             # Input validation
│       └── exporters.py              # Output format handlers
├── configs/
│   ├── basic_config.json             # Basic generation settings
│   ├── advanced_config.json          # Advanced generation settings
│   └── templates/
│       ├── entity_templates.json     # Entity generation templates
│       └── event_templates.json      # Event generation templates
├── examples/
│   ├── quickstart.py                 # Quick start example
│   ├── batch_generation.py           # Batch processing example
│   └── custom_integration.py         # Custom AI integration example
├── tests/
│   ├── test_generators.py            # Unit tests for generators
│   ├── test_evaluation.py            # Unit tests for evaluation
│   └── test_integration.py           # Integration tests
├── docs/
│   ├── api_reference.md              # Complete API documentation
│   ├── configuration_guide.md        # Configuration guide
│   ├── extension_guide.md            # How to extend the system
│   └── examples.md                   # Usage examples
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── README.md                         # Project overview
└── LICENSE                           # License information
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd tax_law_reasoning_generator

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from src.tax_law_generator import TaxLawCaseGenerator, ComplexityLevel
from src.config_evaluation import GenerationConfig

# Initialize generator with default configuration
generator = TaxLawCaseGenerator()

# Generate a single case
case = generator.generate(complexity_level=ComplexityLevel.INTERMEDIATE)

# Access case components
print(f"Case ID: {case.case_id}")
print(f"Narrative: {case.narrative}")
print(f"Ground Truth: {case.ground_truth_answer}")

# Access reasoning chain
for i, step in enumerate(case.reasoning_chain):
    print(f"Step {i+1}: {step.reasoning_text}")
```

### Batch Generation

```python
from src.config_evaluation import DatasetGenerator, GenerationConfig

# Configure generation parameters
config = GenerationConfig(
    complexity_distribution={"basic": 0.3, "intermediate": 0.4, "advanced": 0.3},
    narrative_length_target=500,
    include_distractors=True
)

# Generate dataset
dataset_gen = DatasetGenerator(config)
dataset_info = dataset_gen.generate_dataset(
    num_cases=100,
    output_dir="my_tax_dataset"
)

print(f"Generated {dataset_info['num_cases']} cases")
```

## ⚙️ Configuration

The system uses a flexible configuration system supporting JSON and YAML formats:

```json
{
  "complexity_distribution": {
    "basic": 0.2,
    "intermediate": 0.4,
    "advanced": 0.3,
    "expert": 0.1
  },
  "narrative_length_target": 600,
  "include_distractors": true,
  "tax_year": "2024",
  "jurisdiction": "US_Federal",
  "applicable_tax_codes": ["IRC_61", "IRC_162", "IRC_170"],
  "min_reasoning_steps": 4,
  "max_reasoning_steps": 8
}
```

## 🔧 Extension Points

### Custom Entity Types

```python
class CustomEntityGenerator(EntityGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.entity_templates.update({
            "trust": {
                "attributes": ["trust_type", "beneficiaries", "assets"],
                "complexity_factors": ["distribution_rules", "tax_elections"]
            }
        })
```

### Custom Reasoning Steps

```python
from src.tax_law_generator import ReasoningStep

class AdvancedReasoningStep(ReasoningStep):
    REGULATORY_ANALYSIS = "regulatory_analysis"
    CASE_LAW_APPLICATION = "case_law_application"
    POLICY_CONSIDERATION = "policy_consideration"
```

### AI Provider Integration

```python
from src.config_evaluation import GenerativeAIIntegration

# Custom AI provider
class CustomAIProvider(GenerativeAIIntegration):
    def __init__(self, model_name, api_endpoint):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        
    def enhance_narrative(self, narrative):
        # Custom enhancement logic
        return enhanced_narrative
```

## 📊 Evaluation Framework

The system includes comprehensive evaluation metrics:

- **Quality Metrics**: Narrative coherence, reasoning validity, tax law accuracy
- **Difficulty Metrics**: Complexity estimation, human solvability scores  
- **Structure Metrics**: Entity/event counts, reasoning chain length
- **Performance Metrics**: Generation time, evaluation scores

### Evaluation Example

```python
from src.config_evaluation import CaseEvaluator

evaluator = CaseEvaluator()
metrics = evaluator.evaluate_case(case)

print(f"Overall Quality: {metrics.overall_quality_score:.2f}")
print(f"Estimated Difficulty: {metrics.estimated_difficulty:.2f}")
print(f"Human Solvability: {metrics.human_solvability_score:.2f}")
```

## 🧪 Testing

Run the test suite:

```bash
# All tests
python -m pytest tests/

# Specific test categories
python -m pytest tests/test_generators.py
python -m pytest tests/test_evaluation.py
```

## 📈 Performance Considerations

- **Memory Usage**: Large datasets may require batch processing
- **Generation Speed**: ~1-5 seconds per case depending on complexity
- **AI Enhancement**: Optional but significantly improves quality
- **Parallelization**: Supported for batch generation

## 🔍 Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all requirements are installed
2. **AI API Errors**: Check API keys and rate limits
3. **Configuration Errors**: Validate JSON/YAML syntax
4. **Memory Issues**: Reduce batch size for large datasets

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
generator = TaxLawCaseGenerator(config={"debug_mode": True})
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure tests pass: `python -m pytest`
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/
```

## 📚 Advanced Usage

### Custom Tax Domain

```python
# Define custom tax rules
custom_rules = {
    "international_tax": {
        "transfer_pricing": ["arm_length_principle", "documentation_requirements"],
        "tax_treaties": ["treaty_benefits", "tie_breaker_rules"]
    }
}

# Integrate with generator
config = GenerationConfig()
config.domain_extensions = custom_rules

generator = TaxLawCaseGenerator(config)
```

### Multi-Jurisdictional Cases

```python
config = GenerationConfig(
    jurisdiction="Multi",  
    applicable_jurisdictions=["US_Federal", "CA_Federal", "UK"],
    cross_border_complexity=True
)
```

### Integration with Existing Workflows

```python
# Export to common formats
from src.utils.exporters import DatasetExporter

exporter = DatasetExporter()
exporter.to_huggingface_dataset(cases, "my_tax_dataset")
exporter.to_jsonlines(cases, "cases.jsonl")
exporter.to_csv(evaluation_results, "results.csv")
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the MuSR paper: "Testing the Limits of Chain-of-thought with Multistep Soft Reasoning"
- Inspired by the need for scalable AI evaluation benchmarks
- Built with flexibility for future GenAI evaluation methods
