<div align="center" >
<h1>Synthetic Tax Law Reasoning Data Generator</h1>
</div>
<div align="center">
<em>AI-Powered Synthetic Data Generation to evaluate GenAI reasoning capabilities | Built on MuSR Framework</em>
<br><br>
<strong> Quick Links:</strong> 
<a href="docs/api_reference_guide.md">API Documentation</a> -  
<a href="docs/configuration_guide.md">Configuration</a> -  
<a href="docs/extension_guide.md">Extensions</a> -  
<a href="docs/generated_output_explanation.md">Generated Case Explanation</a> -  
<a href="output/">Outputs</a>
</div>

## Project Overview

This system generates complex tax law scenarios with complete reasoning chains to test the limits of chain-of-thought reasoning in large language models. It addresses the need for scalable, challenging benchmarks that can evolve with advancing AI capabilities.

### Key Features

- **Scalable Complexity**: Generate cases from basic to expert level
- **Realistic Scenarios**: Natural language narratives based on real-world tax situations  
- **Complete Reasoning Chains**: Step-by-step logical progression with ground truth
- **Flexible Architecture**: Easily extensible for new domains and evaluation methods
- **AI Integration**: Built-in support for enhancing content with generative AI
- **Comprehensive Evaluation**: Multi-metric assessment of generated cases

## Project Structure

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
├── docs/
│   ├── api_reference.md              # Complete API documentation
│   ├── configuration_guide.md        # Configuration guide
│   ├── extension_guide.md            # How to extend the system
│   └── generated_output_explanation.md.md   # Explaination of genrated output
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── README.md                         # Project overview
└── LICENSE                           # License information
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ankur-mali/tax-law-reasoning-generator
cd tax-law-reasoning-generator

# Install dependencies
pip install -r requirements.txt

```
## **Chain-of-Thought Implementation Deep Dive**

### **5-Step Structured Reasoning Process**

1. **Fact Identification**: Extract and organize relevant tax facts
2. **Rule Application**: Apply specific IRC sections and tax principles
3. **Calculations**: Detailed mathematical computations with explanations
4. **Interpretation**: Analyze complex scenarios and edge cases
5. **Conclusion**: Synthesize findings into final tax determination
### **AI Enhancement Pipeline**

```python
# AI enhances each reasoning step
reasoning_chain = ai_integration.generate_advanced_reasoning(case)
validation_results = ai_integration.validate_reasoning_chain(case)
complexity_assessment = ai_integration.assess_case_complexity(case)
```

### Quick Demo Usage

```python
from src.tax_law_generator.tax_law_generator import TaxLawCaseGenerator, ComplexityLevel
from src.tax_law_generator.ai_integration import GenerativeAIIntegration, AIConfig, EnhancedTaxLawCaseGenerator

# 1. Basic Generation (Rule-Based)
generator = TaxLawCaseGenerator()
case = generator.generate(ComplexityLevel.EXPERT)
print(f"Generated Case: {case.case_id}")
print(f"Entities: {len(case.entities)} | Events: {len(case.events)}")

# 2. AI-Enhanced Generation (GPT-4 Powered)
ai_config = AIConfig(api_key="your_openai_key", model="gpt-4")
ai_integration = GenerativeAIIntegration(ai_config)
enhanced_generator = EnhancedTaxLawCaseGenerator(generator, ai_integration)

enhanced_case = enhanced_generator.generate(
    ComplexityLevel.EXPERT, 
    use_ai_enhancement=True
)

print(f"AI Validation Score: {enhanced_case.ai_validation_results['overall_confidence_score']}")

```
##  **Environment Setup**

### **Required Environment Variables**
### create a .env file with

```bash
# Set OpenAI API key
OPENAI_API_KEY="your_openai_api_key_here"
```

## **Basic Usage Examples**

### **1. Quick Generation Demo**

Generate a single tax law case to see the system in action:

```bash
# Generate expert-level case
python -m examples.quickstart generate --complexity expert

# Generate intermediate-level case
python -m examples.quickstart generate --complexity intermediate

# Generate with specific configuration
python -m examples.quickstart generate --complexity advanced --config configs/templates/advanced_config.json
```

**What you'll see:**

- Generated case information (ID, entities, events)
- AI validation scores and complexity assessment
- Sample narrative and reasoning chain output
- JSON file saved to `output/` directory

***

## **Batch Generation Examples**

### **2. Large-Scale Dataset Creation**

Perfect for research or evaluation purposes:

```bash
# Generate 100 cases with AI enhancement
python examples/batch_generation.py --cases 100 --output my_dataset --api-key your_openai_key

# Generate 50 cases without AI (faster, but less sophisticated)
python examples/batch_generation.py --cases 50 --no-ai --output basic_dataset

# Use specific configuration template
python examples/batch_generation.py --config basic --cases 25 --output small_dataset

# Generate with custom complexity distribution
python examples/batch_generation.py --cases 200 --config advanced --output research_dataset
```


### **Advanced Batch Options**

```bash
# Analyze existing dataset results
python examples/batch_generation.py --analyze-only --output existing_dataset

# Generate with verbose logging
python examples/batch_generation.py --cases 50 --output verbose_dataset --api-key your_key

# Generate specific complexity level only
python examples/batch_generation.py --cases 30 --complexity expert --output expert_only
```

**Output Structure:**

```
my_dataset/
├── dataset_info.json          # Summary and metadata
├── cases/                     # Individual case files
│   ├── case_001.json
│   ├── case_002.json
│   └── ...
├── evaluation_results.csv     # Quality metrics
└── logs/                      # Generation logs
```


***

## **AI Integration Examples**

### **3. Custom AI Integration Demos**

Explore advanced AI features and customization:

```bash
# Run all AI integration demonstrations
python examples/custom_integration.py --api-key your_openai_key

# Run specific demonstration types
python examples/custom_integration.py --demo basic --api-key your_key
python examples/custom_integration.py --demo multi --api-key your_key
python examples/custom_integration.py --demo validation --api-key your_key

# Multi-provider setup with fallback
python examples/custom_integration.py --demo multi --api-key primary_key --fallback-key backup_key

# Save results to custom file
python examples/custom_integration.py --api-key your_key --output my_ai_results.json
```


### **AI Demo Types:**

| **Demo Type** | **Description** | **Command** |
| :-- | :-- | :-- |
| `basic` | Basic AI enhancement workflow | `--demo basic` |
| `multi` | Multi-provider integration | `--demo multi` |
| `validation` | AI validation workflow | `--demo validation` |
| `all` | All demonstrations (default) | `--demo all` |


***

##  **Environment Setup**

### **Required Environment Variables**

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"
```

##  **Configuration \& Customization**

**Template-Driven Flexibility**: Configure via JSON/YAML in `configs/templates/`

```json
{
  "complexity_distribution": {"expert": 0.6, "advanced": 0.4},
  "ai_enhancement": {
    "model": "gpt-4",
    "reasoning_temperature": 0.3,
    "validation_temperature": 0.1
  },
  "narrative_length_target": 800,
  "include_distractors": true
}
```

**See**: [Configuration Guide](docs/configuration_guide.md) for complete options.

## **Extensibility Features**

### **Add New AI Providers**

```python
class CustomAIProvider(AIProvider):
    def generate_text(self, prompt: str) -> str:
        # Integrate Claude, Gemini, or custom models
        pass
```


### **Custom Entity Types**

```python
# Add to entity_templates.json
{
  "nonprofit_org": {
    "tax_exemption_status": ["501c3", "501c4"],
    "complexity_factors": ["unrelated_business_income"]
  }
}
```

**See**: [Extension Guide](docs/extension_guide.md) for complete examples.

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

## Evaluation Framework

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

## **Documentation \& Resources**

- **[Complete API Reference](docs/api_reference.md)**: Detailed class and method documentation
- **[Configuration Guide](docs/configuration_guide.md)**: Customize generation parameters
- **[Extension Guide](docs/extension_guide.md)**: Add new features and domains
- **[Example Usage](examples/)**: Batch processing, custom integration patterns

***

##  Performance Considerations

- **Memory Usage**: Large datasets may require batch processing
- **Generation Speed**: ~1-5 seconds per case depending on complexity
- **AI Enhancement**: Optional but significantly improves quality
- **Parallelization**: Supported for batch generation

## Troubleshooting

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

## Contributing

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


```

## Advanced Usage

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the MuSR paper: "Testing the Limits of Chain-of-thought with Multistep Soft Reasoning"
- Inspired by the need for scalable AI evaluation benchmarks
- Built with flexibility for future GenAI evaluation methods


------------
## Author
**Ankur Mali**🎓 Master's Student – Engineering Technology & Sustainable Technology Management. Based in Berlin📫 [LinkedIn](https://www.linkedin.com/in/ankur-mali-/) | [GitHub](https://github.com/ankur-mali)
