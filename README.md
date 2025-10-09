# LLM Triage Evaluation

A Python script for testing Large Language Models (LLMs) on patient triage classification using the Manchester Triage System (MTS). This tool evaluates how well different LLMs can classify patients into priority categories (Red, Orange, Yellow, Green, Blue) based on clinical information.

## Overview

The script processes test cases containing patient information and evaluates LLM performance in triaging patients according to the Manchester Triage System. It supports multiple models, validation runs, and generates detailed statistics and logs.

## Requirements

- Python 3.x
- pandas
- ollama (for model querying)
- Required modules in `Modules/` directory

## Usage

```bash
python script.py [OPTIONS]
```

### Command Line Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data` | string | `"test_cases.csv"` | Path to the test cases CSV file containing patient information |
| `--model` | string | `"Todos"` | Model to use for querying. Use "Todos" to test all available models, or specify a specific model name |
| `--validation` | integer | `1` | Number of validation runs per test case (higher values provide more robust results) |
| `--verbose` | integer | `3` | Verbosity level (0-4, higher values show more detailed output) |

### Examples

```bash
# Run with default settings (all models, single validation, default test cases)
python script.py

# Test a specific model with multiple validation runs
python script.py --model "llama3" --validation 3

# Use custom test cases with high verbosity
python script.py --data "my_test_cases.csv" --verbose 4

# Quick test with minimal output
python script.py --model "gpt-3.5-turbo" --validation 1 --verbose 0
```

## Input Data Format

The test cases CSV file should contain the following columns:

- `ID`: Unique identifier for each case
- `Idade`: Patient age
- `Sexo`: Patient gender
- `Fluxograma`: Flowchart category
- `Anamnese`: Medical history/anamnesis
- `Sintomas`: Symptoms
- Vital signs columns (Temperature, Oximetry, Heart Rate, etc.)
- `Classificacao_Correta`: Correct triage classification
- `Justificativa`: Justification for the classification

## Output

The script generates:

1. **Timestamped logs directory** (`./logs/query_log_YYYY-MM-DD_HH-MM/`)
2. **Summary statistics** (`summary_statistics.csv`) with model performance metrics
3. **Detailed results** for each model and validation run

## Prompts

The script uses two built-in prompts:

1. **Instructional Prompt**: Detailed explanation of the MTS system and task
2. **Direct Task Prompt**: Concise classification instruction
