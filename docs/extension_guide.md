# Extension Guide

This guide shows how to extend and customize the Tax Law Reasoning Generator.

## 1. Adding New Entity/Scenario Types

- **Edit Templates:** Add new entity or event types in `configs/templates/entity_templates.json` or `event_templates.json`.
- **Example:**
    ```
    {
      "startup_company": {
        "industry": "AI",
        "revenue_level": ["low", "medium"],
        "typical_deductions": ["R&D_credit", "equipment"]
      }
    }
    ```
- **In Code:** The generators will automatically pick up new types if defined in the templates.

---

## 2. Customizing AI Providers

- Extend `GenerativeAIIntegration` to add another provider (e.g., Anthropic, Gemini).
- Register your own API class in `ai_integration.py`:
    ```
    class CustomAIProvider(AIProvider):
        def generate_text(self, prompt: str, **kwargs):
            # Implement your provider's logic here
            pass
    ```

---

## 3. Adding Custom Reasoning Steps

- Edit `ReasoningChainGenerator` to add new or domain-specific reasoning steps.
- Example: Add a `PENALTY_ASSESSMENT` step for legal fines.
- Remember to update validators to account for new step types.

---

## 4. Exporting to New Formats

- Extend `exporters.py` in `utils`:
    - Add a function like `export_to_xml()` or `export_to_huggingface()`.

---

## 5. Batch/Pipeline Integration

- Use `DatasetGenerator` in custom scripts (see `examples/batch_generation.py`).
- Chain with your preferred data pipeline tools (e.g., Pandas) for further analysis.

---

## 6. Improving Evaluation

- Extend `CaseEvaluator` or add new metrics as needed.
- Metric examples: fairness, surface linguistic style, deeper legal compliance.

---

## 7. Community Extensions

- Submit PRs for new templates, generators, or exporters!
- Extend the documentation in `docs/` if you add major features.

---



---

