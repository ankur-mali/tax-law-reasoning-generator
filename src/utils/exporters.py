"""
Export Module for Tax Law Reasoning Generator
Provides comprehensive export functionality for different research and analysis formats
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
import logging

from .data_structures import (
    TaxLawCase,
    TaxEntity,
    TaxEvent,
    ReasoningStepData,
    EvaluationMetrics,
    DatasetSummary,
    serialize_case
)

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Comprehensive dataset exporter with multiple format support"""

    def __init__(self, output_dir: Union[str, Path] = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def export_dataset(self,
                       cases: List[TaxLawCase],
                       dataset_name: str = "tax_law_dataset",
                       formats: List[str] = None,
                       include_metadata: bool = True,
                       compress: bool = False) -> Dict[str, str]:
        """
        Export dataset in multiple formats

        Args:
            cases: List of tax law cases to export
            dataset_name: Name for the dataset
            formats: List of export formats ['json', 'csv', 'jsonlines', 'xml', 'huggingface']
            include_metadata: Whether to include metadata files
            compress: Whether to compress the output

        Returns:
            Dictionary mapping format names to output file paths
        """
        if formats is None:
            formats = ['json', 'csv', 'jsonlines']

        if not cases:
            raise ValueError("Cannot export empty dataset")

        export_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.output_dir / f"{dataset_name}_{timestamp}"
        dataset_dir.mkdir(exist_ok=True)

        # Export in each requested format
        for format_name in formats:
            try:
                if format_name == 'json':
                    path = self._export_json(cases, dataset_dir / f"{dataset_name}.json")
                elif format_name == 'csv':
                    path = self._export_csv(cases, dataset_dir / f"{dataset_name}.csv")
                elif format_name == 'jsonlines':
                    path = self._export_jsonlines(cases, dataset_dir / f"{dataset_name}.jsonl")
                elif format_name == 'xml':
                    path = self._export_xml(cases, dataset_dir / f"{dataset_name}.xml")
                elif format_name == 'huggingface':
                    path = self._export_huggingface(cases, dataset_dir / "huggingface")
                elif format_name == 'excel':
                    path = self._export_excel(cases, dataset_dir / f"{dataset_name}.xlsx")
                else:
                    logger.warning(f"Unsupported export format: {format_name}")
                    continue

                export_paths[format_name] = str(path)
                logger.info(f"Exported {len(cases)} cases to {format_name}: {path}")

            except Exception as e:
                logger.error(f"Failed to export {format_name}: {e}")

        # Export metadata if requested
        if include_metadata:
            metadata_path = self._export_metadata(cases, dataset_dir / "metadata.json", dataset_name)
            export_paths['metadata'] = str(metadata_path)

        # Compress if requested
        if compress:
            compressed_path = self._compress_dataset(dataset_dir, dataset_name)
            export_paths['compressed'] = str(compressed_path)

        return export_paths

    def _export_json(self, cases: List[TaxLawCase], filepath: Path) -> Path:
        """Export cases as JSON"""
        data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_cases': len(cases),
                'format': 'json'
            },
            'cases': [case.to_dict() for case in cases]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def _export_jsonlines(self, cases: List[TaxLawCase], filepath: Path) -> Path:
        """Export cases as JSON Lines format (one JSON object per line)"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for case in cases:
                json.dump(case.to_dict(), f, ensure_ascii=False)
                f.write('\n')

        return filepath

    def _export_csv(self, cases: List[TaxLawCase], filepath: Path) -> Path:
        """Export cases as CSV with summary information"""
        if not cases:
            return filepath

        fieldnames = [
            'case_id', 'title', 'complexity_level', 'narrative_length',
            'num_entities', 'num_events', 'num_reasoning_steps',
            'total_income', 'total_deductions', 'ground_truth_answer',
            'ai_enhanced', 'ai_validation_score', 'creation_date'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for case in cases:
                writer.writerow({
                    'case_id': case.case_id,
                    'title': case.title,
                    'complexity_level': case.complexity_level.value,
                    'narrative_length': len(case.narrative.split()),
                    'num_entities': len(case.entities),
                    'num_events': len(case.events),
                    'num_reasoning_steps': len(case.reasoning_chain),
                    'total_income': case.calculate_total_income(),
                    'total_deductions': case.calculate_total_deductions(),
                    'ground_truth_answer': case.ground_truth_answer,
                    'ai_enhanced': getattr(case, 'ai_enhanced', False),
                    'ai_validation_score': getattr(case, 'ai_validation_results', {}).get('overall_confidence_score',
                                                                                          ''),
                    'creation_date': case.metadata.get('created_at', '')
                })

        return filepath

    def _export_excel(self, cases: List[TaxLawCase], filepath: Path) -> Path:
        """Export cases as Excel workbook with multiple sheets"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for case in cases:
                summary_data.append({
                    'case_id': case.case_id,
                    'title': case.title,
                    'complexity_level': case.complexity_level.value,
                    'narrative_length': len(case.narrative.split()),
                    'num_entities': len(case.entities),
                    'num_events': len(case.events),
                    'num_reasoning_steps': len(case.reasoning_chain),
                    'ai_enhanced': getattr(case, 'ai_enhanced', False)
                })

            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Entities sheet
            entities_data = []
            for case in cases:
                for entity in case.entities:
                    entities_data.append({
                        'case_id': case.case_id,
                        'entity_id': entity.id,
                        'entity_name': entity.name,
                        'entity_type': entity.entity_type,
                        'attributes': str(entity.attributes)
                    })

            if entities_data:
                pd.DataFrame(entities_data).to_excel(writer, sheet_name='Entities', index=False)

            # Events sheet
            events_data = []
            for case in cases:
                for event in case.events:
                    events_data.append({
                        'case_id': case.case_id,
                        'event_id': event.id,
                        'event_type': event.event_type,
                        'amount': event.amount,
                        'date': event.date,
                        'description': event.description
                    })

            if events_data:
                pd.DataFrame(events_data).to_excel(writer, sheet_name='Events', index=False)

        return filepath

    def _export_xml(self, cases: List[TaxLawCase], filepath: Path) -> Path:
        """Export cases as XML"""
        root = ET.Element('tax_law_dataset')
        root.set('export_timestamp', datetime.now().isoformat())
        root.set('total_cases', str(len(cases)))

        for case in cases:
            case_elem = ET.SubElement(root, 'case')
            case_elem.set('id', case.case_id)
            case_elem.set('complexity', case.complexity_level.value)

            # Basic info
            ET.SubElement(case_elem, 'title').text = case.title
            ET.SubElement(case_elem, 'narrative').text = case.narrative
            ET.SubElement(case_elem, 'ground_truth').text = case.ground_truth_answer

            # Entities
            entities_elem = ET.SubElement(case_elem, 'entities')
            for entity in case.entities:
                entity_elem = ET.SubElement(entities_elem, 'entity')
                entity_elem.set('id', entity.id)
                entity_elem.set('type', entity.entity_type)
                ET.SubElement(entity_elem, 'name').text = entity.name

            # Events
            events_elem = ET.SubElement(case_elem, 'events')
            for event in case.events:
                event_elem = ET.SubElement(events_elem, 'event')
                event_elem.set('id', event.id)
                event_elem.set('type', event.event_type)
                if event.amount:
                    event_elem.set('amount', str(event.amount))
                ET.SubElement(event_elem, 'description').text = event.description

            # Reasoning
            reasoning_elem = ET.SubElement(case_elem, 'reasoning_chain')
            for step in case.reasoning_chain:
                step_elem = ET.SubElement(reasoning_elem, 'step')
                step_elem.set('id', step.step_id)
                step_elem.set('type', step.step_type.value)
                ET.SubElement(step_elem, 'description').text = step.description
                ET.SubElement(step_elem, 'reasoning').text = step.reasoning_text

        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        return filepath

    def _export_huggingface(self, cases: List[TaxLawCase], dirpath: Path) -> Path:
        """Export in HuggingFace datasets format"""
        dirpath.mkdir(exist_ok=True, parents=True)

        # Create dataset info
        dataset_info = {
            "description": "Synthetic Tax Law Reasoning Dataset",
            "citation": "@misc{tax_law_reasoning_2024}",
            "homepage": "",
            "license": "MIT",
            "features": {
                "case_id": {"dtype": "string"},
                "title": {"dtype": "string"},
                "narrative": {"dtype": "string"},
                "complexity_level": {"dtype": "string"},
                "entities": {"dtype": "string"},
                "events": {"dtype": "string"},
                "reasoning_chain": {"dtype": "string"},
                "ground_truth_answer": {"dtype": "string"}
            }
        }

        with open(dirpath / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        # Create training split
        train_data = []
        for case in cases:
            train_data.append({
                "case_id": case.case_id,
                "title": case.title,
                "narrative": case.narrative,
                "complexity_level": case.complexity_level.value,
                "entities": json.dumps([e.to_dict() for e in case.entities]),
                "events": json.dumps([e.to_dict() for e in case.events]),
                "reasoning_chain": json.dumps([s.to_dict() for s in case.reasoning_chain]),
                "ground_truth_answer": case.ground_truth_answer
            })

        with open(dirpath / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)

        return dirpath

    def _export_metadata(self, cases: List[TaxLawCase], filepath: Path, dataset_name: str) -> Path:
        """Export dataset metadata"""
        # Calculate statistics
        complexity_dist = {}
        entity_type_dist = {}
        event_type_dist = {}
        ai_enhanced_count = 0

        for case in cases:
            # Complexity distribution
            complexity = case.complexity_level.value
            complexity_dist[complexity] = complexity_dist.get(complexity, 0) + 1

            # Entity types
            for entity in case.entities:
                entity_type = entity.entity_type
                entity_type_dist[entity_type] = entity_type_dist.get(entity_type, 0) + 1

            # Event types
            for event in case.events:
                event_type = event.event_type
                event_type_dist[event_type] = event_type_dist.get(event_type, 0) + 1

            # AI enhancement
            if getattr(case, 'ai_enhanced', False):
                ai_enhanced_count += 1

        metadata = {
            'dataset_name': dataset_name,
            'export_timestamp': datetime.now().isoformat(),
            'total_cases': len(cases),
            'statistics': {
                'complexity_distribution': complexity_dist,
                'entity_type_distribution': entity_type_dist,
                'event_type_distribution': event_type_dist,
                'ai_enhanced_cases': ai_enhanced_count,
                'ai_enhancement_rate': ai_enhanced_count / len(cases) if cases else 0
            },
            'data_quality': {
                'average_narrative_length': sum(len(c.narrative.split()) for c in cases) / len(cases) if cases else 0,
                'average_entities_per_case': sum(len(c.entities) for c in cases) / len(cases) if cases else 0,
                'average_events_per_case': sum(len(c.events) for c in cases) / len(cases) if cases else 0,
                'average_reasoning_steps': sum(len(c.reasoning_chain) for c in cases) / len(cases) if cases else 0
            },
            'export_info': {
                'generator_version': '1.0.0',
                'export_format_version': '1.0',
                'data_structure_version': '1.0'
            }
        }

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        return filepath

    def _compress_dataset(self, dataset_dir: Path, dataset_name: str) -> Path:
        """Compress dataset directory into ZIP file"""
        zip_path = dataset_dir.parent / f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in dataset_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(dataset_dir)
                    zipf.write(file_path, arcname)

        return zip_path


# Convenience functions for single-format exports
def export_to_json(cases: List[TaxLawCase], filepath: str) -> str:
    """Export cases to JSON format"""
    exporter = DatasetExporter()
    path = exporter._export_json(cases, Path(filepath))
    return str(path)


def export_to_csv(cases: List[TaxLawCase], filepath: str) -> str:
    """Export cases to CSV format"""
    exporter = DatasetExporter()
    path = exporter._export_csv(cases, Path(filepath))
    return str(path)


def export_to_jsonlines(cases: List[TaxLawCase], filepath: str) -> str:
    """Export cases to JSON Lines format"""
    exporter = DatasetExporter()
    path = exporter._export_jsonlines(cases, Path(filepath))
    return str(path)


def export_to_huggingface(cases: List[TaxLawCase], dirpath: str) -> str:
    """Export cases to HuggingFace format"""
    exporter = DatasetExporter()
    path = exporter._export_huggingface(cases, Path(dirpath))
    return str(path)


def export_evaluation_metrics(metrics_list: List[EvaluationMetrics], filepath: str) -> str:
    """Export evaluation metrics to CSV"""
    if not metrics_list:
        return filepath

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'case_id', 'complexity_level', 'num_entities', 'num_events', 'num_reasoning_steps',
            'narrative_length', 'narrative_coherence_score', 'reasoning_validity_score',
            'tax_law_accuracy_score', 'overall_quality_score', 'estimated_difficulty',
            'human_solvability_score', 'ai_enhancement_score', 'ai_validation_score',
            'generation_time', 'evaluation_timestamp'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for metrics in metrics_list:
            writer.writerow(metrics.to_dict())

    return filepath


def export_dataset(cases: List[TaxLawCase],
                   output_dir: str = "exports",
                   dataset_name: str = "tax_law_dataset",
                   formats: List[str] = None) -> Dict[str, str]:
    """Convenience function for dataset export"""
    exporter = DatasetExporter(output_dir)
    return exporter.export_dataset(cases, dataset_name, formats)


# Specialized exporters for research applications
def export_for_model_training(cases: List[TaxLawCase], output_dir: str) -> Dict[str, str]:
    """Export dataset optimized for model training"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    exports = {}

    # Training data in JSONL format
    train_path = output_path / "train.jsonl"
    with open(train_path, 'w') as f:
        for case in cases:
            training_sample = {
                "input": case.narrative,
                "output": case.ground_truth_answer,
                "reasoning": [step.reasoning_text for step in case.reasoning_chain],
                "complexity": case.complexity_level.value,
                "metadata": case.metadata
            }
            json.dump(training_sample, f)
            f.write('\n')

    exports['training_data'] = str(train_path)

    # Evaluation prompts
    eval_path = output_path / "evaluation_prompts.json"
    evaluation_data = []
    for case in cases:
        evaluation_data.append({
            "case_id": case.case_id,
            "prompt": f"Analyze this tax scenario and provide your reasoning:\n\n{case.narrative}",
            "expected_reasoning_steps": len(case.reasoning_chain),
            "ground_truth": case.ground_truth_answer,
            "complexity": case.complexity_level.value
        })

    with open(eval_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)

    exports['evaluation_prompts'] = str(eval_path)

    return exports


def export_for_human_evaluation(cases: List[TaxLawCase], output_dir: str) -> str:
    """Export dataset formatted for human evaluation studies"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create evaluation spreadsheet
    eval_path = output_path / "human_evaluation.xlsx"

    evaluation_data = []
    for case in cases:
        evaluation_data.append({
            'case_id': case.case_id,
            'complexity': case.complexity_level.value,
            'narrative': case.narrative,
            'ground_truth': case.ground_truth_answer,
            'human_rating_accuracy': '',  # To be filled by evaluators
            'human_rating_difficulty': '',  # To be filled by evaluators
            'human_rating_realism': '',  # To be filled by evaluators
            'comments': ''  # To be filled by evaluators
        })

    df = pd.DataFrame(evaluation_data)
    df.to_excel(eval_path, index=False)

    return str(eval_path)
