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


def calculate_confusion_matrix(model_answers, correct_df, labels=None):
	"""
	Returns a dict of confusion matrices per model and prompt.
	"""
	matrices = {}
	for m in model_answers:
		for prompt, cases in m.responses.items():
			y_true = []
			y_pred = []
			for case_id, answers in cases.items():
				correct_answer = correct_df.loc[correct_df['ID'] == case_id, 'Correct'].values[0]
				for ans in answers:
					y_true.append(correct_answer)
					y_pred.append(ans['answer'])
			cm = confusion_matrix(y_true, y_pred, labels=labels)
			matrices[(m.model, prompt)] = cm
	return matrices

def calculate_precision_recall_f1(model_answers, correct_df, labels=None):
	"""
	Returns a DataFrame with precision, recall, and F1 per model and prompt.
	"""
	results = []
	for m in model_answers:
		for prompt, cases in m.responses.items():
			y_true = []
			y_pred = []
			for case_id, answers in cases.items():
				correct_answer = correct_df.loc[correct_df['ID'] == case_id, 'Correct'].values[0]
				for ans in answers:
					y_true.append(correct_answer)
					y_pred.append(ans['answer'])
			prec = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
			rec = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
			f1 = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
			results.append({
				'Model': m.model,
				'Prompt': prompt,
				'Precision': prec,
				'Recall': rec,
				'F1': f1
			})
	return pd.DataFrame(results)

def compare_models_prompts(model_answers, correct_df):
	"""
	Returns a DataFrame comparing accuracy, precision, recall, F1 across models and prompts.
	"""
	acc_df = calculate_accuracy(model_answers, correct_df)
	prf_df = calculate_precision_recall_f1(model_answers, correct_df)
	return pd.merge(acc_df, prf_df, on=['Model', 'Prompt'])

def calculate_error_rate(model_answers, correct_df):
	"""
	Returns a DataFrame with error rate per model and prompt.
	"""
	acc_df = calculate_accuracy(model_answers, correct_df)
	acc_df['ErrorRate'] = 1 - acc_df['Accuracy']
	return acc_df[['Model', 'Prompt', 'ErrorRate']]

def mcnemar_test(model_answers, correct_df):
	"""
	Performs McNemar's test between all pairs of models for each prompt.
	Returns a dict with p-values.
	"""
	results = {}
	for i, m1 in enumerate(model_answers):
		for j, m2 in enumerate(model_answers):
			if i >= j:
				continue
			for prompt in m1.responses.keys():
				if prompt not in m2.responses:
					continue
				y_true = []
				m1_pred = []
				m2_pred = []
				for case_id in m1.responses[prompt].keys():
					correct_answer = correct_df.loc[correct_df['ID'] == case_id, 'Correct'].values[0]
					# Use first answer for each case
					m1_ans = m1.responses[prompt][case_id][0]['answer']
					m2_ans = m2.responses[prompt][case_id][0]['answer']
					y_true.append(correct_answer)
					m1_pred.append(m1_ans)
					m2_pred.append(m2_ans)
				# Build contingency table
				both_correct = sum((a == y and b == y) for a, b, y in zip(m1_pred, m2_pred, y_true))
				m1_only = sum((a == y and b != y) for a, b, y in zip(m1_pred, m2_pred, y_true))
				m2_only = sum((a != y and b == y) for a, b, y in zip(m1_pred, m2_pred, y_true))
				both_wrong = sum((a != y and b != y) for a, b, y in zip(m1_pred, m2_pred, y_true))
				table = [[both_correct, m1_only], [m2_only, both_wrong]]
				try:
					result = mcnemar(table, exact=False)
					p_value = result.pvalue
				except Exception:
					p_value = None
				results[(m1.model, m2.model, prompt)] = p_value
	return results

def prompt_sensitivity(model_answers, correct_df):
	"""
	Returns a DataFrame showing how accuracy varies by prompt for each model.
	"""
	acc_df = calculate_accuracy(model_answers, correct_df)
	pivot = acc_df.pivot(index='Model', columns='Prompt', values='Accuracy')
	return pivot
