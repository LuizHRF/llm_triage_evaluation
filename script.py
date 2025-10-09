import pandas as pd
from Modules.table_processing import process_csv_table, save_results_to_csv
from Modules.querie_exec import query_models
from Modules.prompt_creation import merge_information, add_answering_rules
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Run the triage assessment tool.")
parser.add_argument("--data", type=str, default="extras/test_cases.csv", help="Path to the test cases CSV file.")
parser.add_argument("--model", type=str, default="Todos", help="Model to use for querying.")
parser.add_argument("--validation", type=int, default=1, help="Validation level.")
parser.add_argument("--verbose", type=int, default=3, help="Verbosity level.")
args = parser.parse_args()

data = pd.read_csv(args.data)
info, correct_answers = process_csv_table(data, slices=(2, 6))
prompts = pd.DataFrame([
    {
        "id": 1,
        "name": "Prompt Instrucional",
        "prompt_text": "Você é um sistema inteligente de apoio a decisão clínica. Sua tarefa é classificar pacientes segundo o Protocolo de Triagem de Manchester (MTS), com base nas informações fornecidas sobre a queixa principal, sintomas, sinais vitais e possivelmente uma anamnese. O MTS organiza os pacientes em categorias de prioridade com base em critérios clínicos padronizados, e cada categoria define um tempo máximo recomendado para atendimento. A seguir estão os dados do paciente, identifique a categoria mais apropriada (ex: Vermelho, Laranja, Amarelo, Verde ou Azul) e justifique a escolha com base nos critérios do MTS.",
        "created_at": "2024-06-01"
    },
    {
        "id": 2,
        "name": "Prompt de tarrefa direta",
        "prompt_text": "Classifique pacientes segundo o Protocolo de Triagem de Manchester (MTS) com base nas informações fornecidas. Indique a categoria mais apropriada (Vermelho, Laranja, Amarelo, Verde ou Azul) e justifique a escolha.",
        "created_at": "2024-06-02"
    },
])

test_cases_prompts = merge_information(prompts, info)
test_cases_prompts = add_answering_rules(test_cases_prompts)

model_results = query_models(test_cases_prompts, 
                       validation=args.validation, 
                       model=args.model,
                       verbose=args.verbose
                       )

save_results_to_csv(model_results, correct_answers=correct_answers)
