import pandas as pd
import argparse

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser(description="Run the triage assessment tool.")
parser.add_argument("--validation", type=int, default=1, help="Validation level")
parser.add_argument("--data", type=str, help="Path to the test cases CSV file.")
parser.add_argument("--results_path", type=str, help="Path to the results CSV file.")
parser.add_argument("--results_filename", type=str, help="Name of the results CSV file.")
parser.add_argument("--prompts-used", type=int, default=1, help="Number of prompts used")
parser.add_argument("--model", type=str, help="Model name")
args = parser.parse_args()

def get_case_responses(results, prompts_used, validation, correct_answers: pd.DataFrame = None):
    ids = results['ID']
    results = results.drop(columns=['ID'])
    prompt_results =[]

    for i in range(prompts_used):
        prompt_answers = results.iloc[:, i*validation:i*validation+validation]
        prompt_name = prompt_answers.columns[0][:-4]
        #prompt_answers[f"{prompt_name}(moda)"] = prompt_answers.mode(axis=1)[0]
        prompt_answers.loc[:, f"{prompt_name}(moda)"] = prompt_answers.mode(axis=1)[0]
        ids = pd.concat([ids, prompt_answers[f"{prompt_name}(moda)"]], axis=1)
        results_mode= pd.concat([ids["ID"], prompt_answers], axis=1)

        if correct_answers is not None:
            results_mode = pd.merge(results_mode, correct_answers, on="ID", how="left")

        prompt_results.append(results_mode)
    return ids, prompt_results

def calculate_metrics(model, results, prompts_used, validation, correct_df: pd.DataFrame) -> pd.DataFrame:
	
    accuracy_results = []

    results_, _ = get_case_responses(results, prompts_used, validation)

    for prompt in range(prompts_used):
        y_true = correct_df['Classificacao_Correta'].values
        y_pred = results_.iloc[:, prompt+1].values
        prompt_name = results_.columns[prompt+1][:-7]

        #print(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        #conf_matrix = confusion_matrix(y_true, y_pred)

        accuracy_results.append({
            'Model': model,
            'Prompt':prompt_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
        })

    return pd.DataFrame(accuracy_results)
try:
    table = pd.read_csv(f'{args.results_path}/{args.results_filename}')
    data = pd.read_csv(args.data)[["ID","Classificacao_Correta"]]

    x, y =get_case_responses(table, args.prompts_used, args.validation)
    a = pd.concat([x, data["Classificacao_Correta"]], axis=1)
    a.to_csv(f"{args.results_path}/results_summary.csv", index=False)

    summary= calculate_metrics(args.model, table, args.prompts_used, args.validation, data)
    print(summary)
    summary.to_csv(f"statistics.csv", index=False)
    print("Métricas calculadas e salvas com sucesso.")
except Exception as e:
    print(f"Erro ao calcular métricas: {e}")
