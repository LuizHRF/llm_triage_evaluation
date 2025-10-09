from networkx import display
import pandas as pd
from Modules.model_answer import modelAnswer
import os

from Modules.statistics import calculate_metrics
import re

def process_csv_table(data: pd.DataFrame, slices: tuple = None):
    """
    Processes a DataFrame by optionally slicing rows, separating the last two columns as results, and returning both the main features and results DataFrames.

    Args:
        data (pd.DataFrame): Input DataFrame.
        slices (list, optional): List of row indices to select. Defaults to None.

    Returns:
        tuple: (information, results_df)
            - information: DataFrame with all columns except the last two.
            - results_df: DataFrame with 'ID' and the last two columns.
    """

    if slices:
        data = data.iloc[slices[0]:slices[1], :]
        

    information = data.iloc[:, :-2].reset_index(drop=True)
    results_df = data.iloc[:, -2:].reset_index(drop=True)
    results_df = pd.concat([information["ID"], results_df], axis=1)

    return information, results_df

def save_results_to_csv(results: list[modelAnswer], correct_answers: pd.DataFrame = None):
    """
    Saves the results DataFrame to a CSV file.

    Args:
        results (list[modelAnswer]): List of modelAnswer objects containing the results to be saved.
    """
    
    if not results:
        raise ValueError("The results list is empty.")
    
    day = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = f"./logs/query_log_{day}"
    os.makedirs(log_dir, exist_ok=True)

    summary= calculate_metrics(results, correct_answers)
    summary.to_csv(f"{log_dir}/summary_statistics.csv", index=False)

    for r in results:

        # Replace potentially problematic characters for folder names
        modelo = re.sub(r'[<>:"/\\|?*]', '_', r.model)

        folder = f"{log_dir}/{modelo}"
        os.makedirs(folder, exist_ok=True)

        a = r.build_responses_df()
        a.to_csv(f"{folder}/full_response.csv", index=False)

        b, _ = r.get_case_responses()
        results_mode = pd.merge(b, correct_answers, on="ID", how="left").drop(columns=["Justificativa"])
        results_mode.to_csv(f"{folder}/results_summary.csv", index=False)
