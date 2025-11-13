import pandas as pd
import argparse

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

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

def calcular_concordancia(respostas):
    arr = np.asarray(respostas)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("respostas must be a 2D array with 3 columns")
    a, b, c = arr[:, 0], arr[:, 1], arr[:, 2]
    eq_counts = (a == b).astype(int) + (b == c).astype(int) + (a == c).astype(int)
    per_row_agreement = eq_counts / 3.0
    return float(per_row_agreement.mean())

def replace_colors(value):
    color_map = {
        'Vermelho': 5,
        'Laranja': 4,
        'Amarelo': 3,
        'Verde': 2,
        'Azul': 1,
        'vermelho': 5,
        'laranja': 4,
        'amarelo': 3,
        'verde': 2,
        'azul': 1,
        'Blue': 1,
        'Green': 2,
        'Yellow': 3,
        'Orange': 4,
        'Red': 5
        
    }
    return color_map[value]

def calcula_acuracia_geral_por_prompt(y_true, y_pred):

    arr = np.asarray(y_pred)
    a, b, c = arr[:, 0], arr[:, 1], arr[:, 2]

    correct_counts = ((a == y_true).astype(int) + (b == y_true).astype(int) + (c == y_true).astype(int))
    total_predictions = arr.shape[0] * 3
    accuracy = correct_counts.sum() / total_predictions
    return accuracy

def calcula_under_over_triage_geral(y_true, y_pred):

    vfunc = np.vectorize(replace_colors)

    arr = np.asarray(y_pred)
    a, b, c = vfunc(arr[:, 0]), vfunc(arr[:, 1]), vfunc(arr[:, 2])
    y_true_mapped = vfunc(np.asarray(y_true))

    under_triage_counts = (((a < y_true_mapped).astype(int) + (b < y_true_mapped).astype(int) + (c < y_true_mapped).astype(int)))
    over_triage_counts = (((a > y_true_mapped).astype(int) + (b > y_true_mapped).astype(int) + (c > y_true_mapped).astype(int)))
    total_miss_predictions = (((a != y_true_mapped).astype(int) + (b != y_true_mapped).astype(int) + (c != y_true_mapped).astype(int))).sum()
    under_triage_rate = under_triage_counts.sum() / total_miss_predictions
    over_triage_rate = over_triage_counts.sum() / total_miss_predictions
    return under_triage_rate, over_triage_rate

def calcula_under_over_triage_mode(y_true, y_pred):

    vfunc = np.vectorize(replace_colors)
    arr = np.asarray(y_pred)
    y_pred_mapped = vfunc(arr)
    y_true_mapped = vfunc(np.asarray(y_true))

    under_triage_counts = (((y_pred_mapped < y_true_mapped).astype(int)).sum())
    over_triage_counts = (((y_pred_mapped > y_true_mapped).astype(int)).sum())
    total_miss_predictions = (((y_pred_mapped != y_true_mapped).astype(int)).sum())
    under_triage_rate = under_triage_counts / total_miss_predictions
    over_triage_rate = over_triage_counts / total_miss_predictions
    return under_triage_rate, over_triage_rate


def calculate_metrics(model, results, prompts_used, validation, correct_df: pd.DataFrame) -> pd.DataFrame:
	
    accuracy_results = []

    results_, _ = get_case_responses(results, prompts_used, validation)

    for prompt in range(prompts_used):

        #Estatísticas gerais
        prompt_name = results_.columns[prompt+1][:-7]
        y_true = correct_df['Classificacao_Correta'].values

        accuracy_geral = calcula_acuracia_geral_por_prompt(y_true, results.iloc[:, (prompt*3)+1:(prompt*3)+4].values)
        under_triage_rate, over_triage_rate = calcula_under_over_triage_geral(y_true, results.iloc[:, (prompt*3)+1:(prompt*3)+4].values)

        # Estatísticas por moda
        y_pred = results_.iloc[:, prompt+1].values
        
        n = results.iloc[:, (prompt*3)+1:(prompt*3)+4].values
        concordancia = calcular_concordancia(n)

        under_triage_mode, over_triage_mode = calcula_under_over_triage_mode(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        #conf_matrix = confusion_matrix(y_true, y_pred)

        accuracy_results.append({
            'Model': model,
            'Prompt':prompt_name,
            'Acurácia (moda)': accuracy,
            'Precisão (moda)': precision,
            'Recall (moda)': recall,
            'F1 (moda)': f1,
            'Under-triage (moda)': under_triage_mode,
            'Over-triage (moda)': over_triage_mode,
            'Concordância média': concordancia,
            'Acurácia geral': accuracy_geral,
            'Under-triage geral': under_triage_rate,
            'Over-triage geral': over_triage_rate
        })

    results = pd.DataFrame(accuracy_results)
    # add a total row averaging numeric/result columns
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        means = results[numeric_cols].mean()
        total_row = means.to_dict()
        total_row['Model'] = model
        total_row['Prompt'] = 'Total'
        results = pd.concat([results, pd.DataFrame([total_row])], ignore_index=True, sort=False)

    return results


#try:
table = pd.read_csv(f'{args.results_path}/{args.results_filename}')
data = pd.read_csv(args.data)[["ID","Classificacao_Correta"]]

x, y =get_case_responses(table, args.prompts_used, args.validation)
a = pd.concat([x, data["Classificacao_Correta"]], axis=1)
a.to_csv(f"{args.results_path}/results_summary.csv", index=False)

summary= calculate_metrics(args.model, table, args.prompts_used, args.validation, data)

print(summary)

summary.to_csv(f"statistics.csv", index=False)

summary = summary.round(2)
summary.to_csv(f"statistics_pretty.csv", index=False)
print("Métricas calculadas e salvas com sucesso.")
#except Exception as e:
    #print(f"Erro ao calcular métricas: {e}")


# uv run cal_statistics.py --data test_cases_new.csv --results_path ./ --results_filename full_responses.csv --model deepsseek --prompts-used 3 --validation 3