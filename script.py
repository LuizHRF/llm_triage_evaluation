
# uv run python script.py --models "gemma3:4b" "deepseek-r1:1.5b" --data "test_cases_new.csv" --validation 3 --prompts 1 2 --path_to_save "./novos_logs" --check-progress 

import pandas as pd
from Modules.table_processing import process_csv_table, save_results_to_csv
from Modules.querie_exec import query_models, get_ollama_models
from Modules.prompt_creation import add_document_references, merge_information, add_answering_rules
from Modules.statistics import calculate_metrics
import argparse
import os
from tqdm import tqdm
import time
from Modules.rag import rag_agent

def check_params(args):
    if args.validation < 1:
        raise ValueError("Validation level must be at least 1.")
    if args.verbose < 0 or args.verbose > 4:
        raise ValueError("Verbosity level must be between 0 and 4.")
    available_models = get_ollama_models()
    for model in args.models:
        if model not in available_models and model != "Todos":
            raise ValueError(f"Model {model} is not available. Choose from {available_models} or 'Todos'.")
        
def progress_check(current_case, total_cases, messages):
    global _progress_bar, _progress_start_time
    os.system("cls" if os.name == "nt" else "clear")

    try:
        _progress_bar
    except NameError:
        _progress_bar = None
    try:
        _progress_start_time
    except NameError:
        _progress_start_time = None

    # Close and recreate if total changed (to avoid mismatched totals)
    if _progress_bar is None or getattr(_progress_bar, "total", None) != total_cases:
        if _progress_bar is not None:
            try:
                _progress_bar.close()
            except Exception:
                pass
        _progress_bar = tqdm(
            total=total_cases,
            desc="Processing",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}]",
            position=0,
            leave=True,
            miniters=1,
        )
        _progress_start_time = time.time()

    # Update elapsed time and render the progress bar first so it stays above subsequent prints
    elapsed = time.time() - _progress_start_time if _progress_start_time is not None else 0
    _progress_bar.n = current_case - 1
    _progress_bar.refresh()

    # Print messages using plain print so they appear below the bar
    for message in messages:
        print(str(message), flush=True)

# Argument parsing
parser = argparse.ArgumentParser(description="Run the triage assessment tool.")
parser.add_argument("--data", type=str, default="test_cases.csv", help="Path to the test cases CSV file.")
parser.add_argument('--models', nargs='+', type=str, default=None, help="Model to use for querying.")
parser.add_argument("--validation", type=int, default=1, help="Validation level.")
parser.add_argument("--verbose", type=int, default=3, help="Verbosity level.")
parser.add_argument("--prompts", nargs= '+', type= int, default = 0, help="Prompt to be used (ID). 0 for all prompts")
parser.add_argument("--path_to_save", type=str, default='./logs', help="Path to save the responses.")
parser.add_argument("--check-progress", action="store_true", help="Enable progress checking.")
parser.add_argument("--rag", action="store_true", help="Enable RAG functionality.")
args = parser.parse_args()

check_params(args)

data = pd.read_csv(args.data)
info, correct_answers = process_csv_table(data)

print("Iniciando processamento...")
prompts = pd.DataFrame([
    {
        "id": 1,
        "name": "Prompt Instrucional (role)",
        "prompt_text": "Você é um sistema inteligente de apoio a decisão clínica. Sua tarefa é classificar pacientes segundo o Protocolo de Triagem de Manchester (MTS), com base nas informações fornecidas sobre a queixa principal, sintomas, sinais vitais e possivelmente uma anamnese. O MTS organiza os pacientes em categorias de prioridade com base em critérios clínicos padronizados, e cada categoria define um tempo máximo recomendado para atendimento. A seguir estão os dados do paciente, identifique a categoria mais apropriada (ex: Vermelho, Laranja, Amarelo, Verde ou Azul) e justifique a escolha com base nos critérios do MTS.\nAqui estão as informações do paciente:",
        "created_at": "2024-06-01"
    },
    {
        "id": 2,
        "name": "Prompt de tarrefa direta",
        "prompt_text": "Classifique pacientes segundo o Protocolo de Triagem de Manchester (MTS) com base nas informações fornecidas. Indique a categoria mais apropriada (Vermelho, Laranja, Amarelo, Verde ou Azul) e justifique a escolha.",
        "created_at": "2024-06-02"
    },
    {
        "id": 3,
        "name": "Prompt Chain-of-thought",
        "prompt_text": "Você é um sistema inteligente de apoio a decisão clínica com expertise no Protocolo de Triagem de Manchester (MTS). Sua tarefa é classificar um paciente, seguindo um processo de raciocínio lógico em três etapas (Chain-of-Thought), antes de fornecer a classificação final. Instruções (Chain-of-Thought - CoT): 1. Análise da Queixa Principal e Sintomas: Identifique a Queixa Principal e os Sintomas e Sinais Vitais críticos fornecidos, correlacionando-os com os Discriminadores de Risco do MTS. Mencione qual Categoria de Risco Inicial (ex: Risco Imediato, Muito Urgente, Urgente) você está considerando neste ponto. 2. Determinação do Fluxograma e do Discriminador-Chave: Mencione qual Fluxograma do MTS é o mais apropriado para a queixa (ex: Dor, Problemas Respiratórios, Trauma). Em seguida, identifique o Discriminador-Chave (a pergunta ou critério que define a categoria) que leva à sua escolha. 3. Conclusão Justificada (MTS): Com base nos passos 1 e 2, defina a Categoria de Prioridade (Vermelho, Laranja, Amarelo, Verde ou Azul) e o Tempo Máximo de Atendimento recomendado. Resposta Final: Apresente os três passos de raciocínio (CoT) e, por fim, a Categoria Final. A seguir estão as informações do paciente:",
        "created_at": "2024-06-02"
    },
])

if (args.prompts):
    prompts = prompts[prompts["id"].isin(args.prompts)]

if args.rag:
    try:
        rag_agent_instance = rag_agent(protocol_text=open("protocolo_splits.txt", "r", encoding="utf-8").read())
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RAG agent: {e}")

test_cases_prompts, patient_info = merge_information(prompts, info)

# Adicionar RAG
if args.rag:
    test_cases_prompts = add_document_references(test_cases_prompts, rag_agent_instance, patient_info)

test_cases_prompts = add_answering_rules(test_cases_prompts)

test_cases_prompts.to_csv("test_cases_prompts_final.csv", index=False)

# verbose= args.verbose
# if args.check_progress:
#     query_models.progress_callback = progress_check
#     verbose=0

# model_results = query_models(test_cases_prompts, 
#                        validation=args.validation, 
#                        model=args.models,
#                        verbose=verbose,
#                        path_to_save=args.path_to_save
#                        )

# save_results_to_csv(model_results, correct_answers=correct_answers, path=args.path_to_save)

# summary= calculate_metrics(model_results, correct_answers)
# print(summary)


