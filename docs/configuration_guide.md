# Configuration Guide

This guide explains how to configure the Tax Law Reasoning Generator for your needs.

## 1. Overview

The system is driven by configuration files in JSON or YAML format, found in `configs/templates/`. These files define:
- **Case complexity levels**
- **How many entities or events per case**
- **Available tax codes**
- **Narrative styles and output preferences**

Most users only need to update a config template and point the generator to it.

---

## 2. Configuration Structure

### Example: `basic_config.json`
```json
{
"complexity_distribution": {"basic": 0.5, "intermediate": 0.3, "advanced": 0.2},
"min_entities_per_case": 1,
"max_entities_per_case": 2,
"tax_year": "2024",
"narrative_length_target": 300,
"include_distractors": false
}
```

### Key Fields

| Field                      | Type      | Description                                       |
|----------------------------|-----------|---------------------------------------------------|
| complexity_distribution    | object    | Fraction for each complexity (sum = 1.0)           |
| min_entities_per_case      | integer   | Minimum entities (people, businesses) per case     |
| max_entities_per_case      | integer   | Maximum entities per case                          |
| min_events_per_case        | integer   | Minimum number of events (income, credits, etc.)   |
| max_events_per_case        | integer   | Maximum number of events                           |
| tax_year                   | string    | Tax year for scenario (e.g., "2024")               |
| applicable_tax_codes       | list      | IRC/other codes to include                         |
| narrative_length_target    | integer   | Target length for output narratives (words)        |
| include_distractors        | bool      | Add irrelevant facts for increased challenge       |
| output_format              | string    | One of: 'json', 'csv', 'yaml'                     |

---

## 3. Template Usage

- **Location:** Place your config file under `configs/templates/`
- **Naming:** Use a descriptive filename, e.g. `my_study_config.json`
- **Loading:** Use in code:
    ```
    from src.tax_law_generator.config_evaluation import ConfigManager
    config = ConfigManager().load_config('configs/templates/my_study_config.json')
    ```

---

## 4. Creating Your Own Config

1. Copy an existing JSON or YAML template.
2. Adjust entity, event, and complexity settings.
3. Save with a new filename in `configs/templates/`.

---

## 5. Tips

- Keep `complexity_distribution` summing to 1.0.
- For faster tests, use lower `narrative_length_target`.
- To simulate real-world noise, set `include_distractors` to true.

---

## 6. Troubleshooting

- **File not found:** Ensure the path and extension match exactly.
- **Validation errors:** Check for typos, missing fields, or invalid data types.

See the project’s README or API docs for troubleshooting help and further configuration options.
