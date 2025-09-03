#!/usr/bin/env python3
"""
Basic Test Script - Verify the tax law generator is working correctly
Run this to test your implementation step by step
"""

import sys
import os
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test 1: Verify all imports work"""
    print("🧪 Test 1: Testing imports...")
    try:
        from tax_law_generator.tax_law_generator import (
            TaxLawCaseGenerator,
            ComplexityLevel,
            EntityGenerator,
            EventGenerator,
            NarrativeGenerator,
            ReasoningChainGenerator
        )
        print("✅ Core imports successful")

        from tax_law_generator.config_evaluation import (
            GenerationConfig,
            CaseEvaluator,
            DatasetGenerator
        )
        print("✅ Configuration imports successful")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n🔧 Fix: Check that files are in correct locations:")
        print("  - src/tax_law_generator/tax_law_generator.py")
        print("  - src/tax_law_generator/config_evaluation.py")
        print("  - src/tax_law_generator/__init__.py")
        return False


def test_entity_generation():
    """Test 2: Verify entity generation"""
    print("\n🧪 Test 2: Testing entity generation...")
    try:
        from tax_law_generator.tax_law_generator import EntityGenerator

        entity_gen = EntityGenerator()

        # Test individual entity
        individual = entity_gen.generate("individual")
        print(f"✅ Generated individual: {individual.name}")
        print(f"   Attributes: {individual.attributes}")

        # Test corporation entity
        corporation = entity_gen.generate("corporation")
        print(f"✅ Generated corporation: {corporation.name}")
        print(f"   Attributes: {corporation.attributes}")

        return True

    except Exception as e:
        print(f"❌ Entity generation failed: {e}")
        return False


def test_event_generation():
    """Test 3: Verify event generation"""
    print("\n🧪 Test 3: Testing event generation...")
    try:
        from tax_law_generator.tax_law_generator import EntityGenerator, EventGenerator

        # Create entities for events
        entity_gen = EntityGenerator()
        entities = [entity_gen.generate("individual")]

        # Generate events
        event_gen = EventGenerator()

        income_event = event_gen.generate("income", entities)
        print(f"✅ Generated income event: {income_event.event_type}")
        print(f"   Amount: ${income_event.amount:,.2f}")

        deduction_event = event_gen.generate("deduction", entities)
        print(f"✅ Generated deduction event: {deduction_event.event_type}")
        print(f"   Amount: ${deduction_event.amount:,.2f}")

        return True

    except Exception as e:
        print(f"❌ Event generation failed: {e}")
        return False


def test_case_generation():
    """Test 4: Verify complete case generation"""
    print("\n🧪 Test 4: Testing complete case generation...")
    try:
        from tax_law_generator.tax_law_generator import TaxLawCaseGenerator, ComplexityLevel

        generator = TaxLawCaseGenerator()

        # Test each complexity level
        for complexity in ComplexityLevel:
            print(f"  Testing {complexity.value} complexity...")
            case = generator.generate(complexity_level=complexity)

            print(f"    ✅ Case ID: {case.case_id}")
            print(f"    ✅ Entities: {len(case.entities)}")
            print(f"    ✅ Events: {len(case.events)}")
            print(f"    ✅ Reasoning steps: {len(case.reasoning_chain)}")
            print(f"    ✅ Narrative length: {len(case.narrative.split())} words")

            # Verify narrative is not empty
            if not case.narrative or len(case.narrative) < 50:
                print(f"    ⚠️  Warning: Narrative seems too short")

            # Verify ground truth exists
            if not case.ground_truth_answer:
                print(f"    ⚠️  Warning: No ground truth answer")

        print("✅ All complexity levels generated successfully")
        return True

    except Exception as e:
        print(f"❌ Case generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test 5: Verify evaluation system"""
    print("\n🧪 Test 5: Testing evaluation system...")
    try:
        from tax_law_generator.tax_law_generator import TaxLawCaseGenerator, ComplexityLevel
        from tax_law_generator.config_evaluation import CaseEvaluator, GenerationConfig

        # Generate a test case
        generator = TaxLawCaseGenerator()
        case = generator.generate(complexity_level=ComplexityLevel.INTERMEDIATE)

        # Evaluate the case
        evaluator = CaseEvaluator()
        metrics = evaluator.evaluate_case(case)

        print(f"✅ Evaluation completed for case: {case.case_id}")
        print(f"   Overall Quality Score: {metrics.overall_quality_score:.3f}")
        print(f"   Narrative Coherence: {metrics.narrative_coherence_score:.3f}")
        print(f"   Reasoning Validity: {metrics.reasoning_validity_score:.3f}")
        print(f"   Tax Law Accuracy: {metrics.tax_law_accuracy_score:.3f}")
        print(f"   Estimated Difficulty: {metrics.estimated_difficulty:.3f}")

        return True

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_generation():
    """Test 6: Verify batch dataset generation"""
    print("\n🧪 Test 6: Testing dataset generation...")
    try:
        from tax_law_generator.config_evaluation import DatasetGenerator, GenerationConfig

        # Create simple config
        config = GenerationConfig(
            complexity_distribution={"basic": 0.5, "intermediate": 0.5},
            narrative_length_target=300
        )

        # Generate small dataset
        dataset_gen = DatasetGenerator(config)
        dataset_info = dataset_gen.generate_dataset(
            num_cases=3,
            output_dir="test_output"
        )

        print(f"✅ Generated dataset with {dataset_info['num_cases']} cases")
        print(f"   Output directory: test_output/")
        print(f"   Average quality: {dataset_info['evaluation_summary']['average_quality_score']:.3f}")

        # Verify files were created
        output_path = Path("test_output")
        if output_path.exists():
            files = list(output_path.rglob("*"))
            print(f"   Created {len(files)} files")

        return True

    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_sample_case():
    """Save a sample case for manual inspection"""
    print("\n📄 Saving sample case for inspection...")
    try:
        from tax_law_generator.tax_law_generator import TaxLawCaseGenerator, ComplexityLevel

        generator = TaxLawCaseGenerator()
        case = generator.generate(complexity_level=ComplexityLevel.INTERMEDIATE)

        # Create readable output
        sample_output = {
            "case_info": {
                "case_id": case.case_id,
                "title": case.title,
                "complexity": case.complexity_level.value
            },
            "narrative": case.narrative,
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "attributes": e.attributes
                } for e in case.entities
            ],
            "events": [
                {
                    "type": e.event_type,
                    "amount": e.amount,
                    "description": e.description
                } for e in case.events
            ],
            "reasoning_chain": [
                {
                    "step": i + 1,
                    "type": step.step_type.value,
                    "description": step.description,
                    "reasoning": step.reasoning_text
                } for i, step in enumerate(case.reasoning_chain)
            ],
            "ground_truth": case.ground_truth_answer
        }

        # Save to file
        with open("sample_case.json", "w") as f:
            json.dump(sample_output, f, indent=2)

        print(f"✅ Sample case saved to: sample_case.json")
        print(f"   Case ID: {case.case_id}")
        print(f"   Narrative: {case.narrative[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Failed to save sample case: {e}")
        return False


def run_all_tests():
    """Run all tests in sequence"""
    print("🚀 Starting Tax Law Reasoning Generator Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_entity_generation,
        test_event_generation,
        test_case_generation,
        test_evaluation,
        test_dataset_generation,
        save_sample_case
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        if not result:
            print(f"\n❌ Test failed: {test.__name__}")
            print("   Fix this issue before proceeding to next test")
            break
        print("   ✅ PASSED\n")

    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("\n✅ Your tax law reasoning generator is working correctly!")
        print("\n📁 Check these files:")
        print("   - sample_case.json (example generated case)")
        print("   - test_output/ (sample dataset)")

        print("\n🚀 Next steps:")
        print("   1. Run: python examples/quickstart.py")
        print("   2. Customize configs/basic_config.json")
        print("   3. Generate larger datasets")

    else:
        print(f"❌ {total - passed} tests failed. Fix issues above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)