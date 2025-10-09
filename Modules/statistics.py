import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from Modules.model_answer import modelAnswer
from statsmodels.stats.contingency_tables import mcnemar

def calculate_metrics(model_answers: list[modelAnswer], correct_df: pd.DataFrame) -> pd.DataFrame:
	
	accuracy_results = []

	for model in model_answers:
		results, _ = model.get_case_responses()
		#print(results)
		for prompt in results.columns[1:]:
			y_true = correct_df['Classificacao_Correta'].values
			y_pred = results[prompt].values

			#print(y_true, y_pred)

			accuracy = accuracy_score(y_true, y_pred)
			
			precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
			recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
			f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
			#conf_matrix = confusion_matrix(y_true, y_pred)

			accuracy_results.append({
				'Model': model.model,
				'Prompt': prompt[:-7],
				'Accuracy': accuracy,
				'Precision': precision,
				'Recal': recall,
				'F1': f1,
			})
	return pd.DataFrame(accuracy_results)
