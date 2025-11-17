import pandas as pd

def merge_information(prompts: pd.DataFrame, 
                      test_cases_df: pd.DataFrame,
                      prompts_to_use: list[int] = None
                      ) -> pd.DataFrame:

    case_ids = test_cases_df["ID"]
    test_cases_df = test_cases_df.drop(columns=["ID"])

    patient_info = ("".join([f"{col}: {row[col]}\n" for col in test_cases_df.columns if pd.notna(row[col])]) for _, row in test_cases_df.iterrows())
    patient_info = pd.Series(patient_info)

    result_df = pd.DataFrame()
    for _, prompt_row in prompts.iterrows():
        prompt_text = prompt_row["prompt_text"]
        prompt_col = f"prompt_{prompt_row['id']}"

        result_df[prompt_col] = [
            f"{prompt_text} \n" +
            "".join(patient_info[row]) for row in range(len(patient_info))] 
        
    return pd.concat([case_ids, result_df], axis=1), patient_info

def add_answering_rules(prompts: pd.DataFrame):
    """
        Adds answering rules to each prompt in the DataFrame.
        Args:
            prompts (pd.DataFrame): DataFrame containing prompts.
        Returns:
            pd.DataFrame: DataFrame with updated prompts including answering rules.
    """

#     rules = """\nCom base nessas informacoes, qual e a cor de atendimento do paciente?
# Nao utilize emojis ou outros caracteres na resposta e escreva a explicacao em texto corrido.
# Responda no formato json: {"resposta": "r", "explicacao": "e"}"""

    rules = """\nCom base nessas informacoes, qual e a cor de atendimento do paciente?
Nao utilize emojis ou outros caracteres na resposta e escreva a explicacao em texto corrido.
Responda **apenas** com JSON puro, sem markdown, sem texto antes ou depois. Exemplo de saída esperada: {"resposta":"...", "explicacao":"..."}
Responda no formato json, com as chaves "resposta" e "explicacao"."""
 
    for col in prompts.columns[1:]:
        prompts[col] = prompts[col].astype(str) + "\n" + rules

    return prompts

def add_document_references(prompts: pd.DataFrame, rag_agent_instance, patient_info: pd.Series):
    
    for col in prompts.columns[1:]:
        for i in range(len(prompts)):
            prompts.at[i, col] = rag_agent_instance.improve_query(prompts.at[i, col], patient_info[i])
    return prompts