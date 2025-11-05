import os
import ollama
import pandas as pd
import json
from Modules.model_answer import modelAnswer
import re

def get_ollama_models():
    models = []
    for i in ollama.list():
        for model in i[1]:
            models.append(model.model)
    return models

def fix_response(response: str) -> str:

    """
        Attempts to fix common JSON formatting issues and other inconsistencies in the response string.
    Args:
        response (str): The raw response string from the model.
    Returns:
        str: The corrected response string.
    """

    # Remove Markdown code blocks
    text = response.strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    return text

def query_models(prompts: pd.DataFrame, 
                 validation: int = 1, 
                 model = None, 
                 verbose: int = 0,
                 path_to_save: str = None,
                 ) -> list[modelAnswer]:

    if model is None or model == "Todos":
        models = get_ollama_models()
    else:
        models = model

    total_queries = len(models) * (prompts.shape[1] - 1) * prompts.shape[0] * validation
    current_query = 0
    messages = []

    def update_progress_bar():
        nonlocal current_query
        current_query += 1

        if hasattr(query_models, "progress_callback") and callable(query_models.progress_callback):
            query_models.progress_callback(current_query, total_queries, messages)

    update_progress_bar()
    responses = []

    for model in models:
        if path_to_save:
            os.makedirs(f'{path_to_save}/{model}/', exist_ok=True)
            saving_df = pd.DataFrame(columns=["ID"] + [f'{col} ({i+1}x)' for col in prompts.columns if col != 'ID' for i in range(validation)] + [f'{col} ({i+1}x) Explanation' for col in prompts.columns if col != 'ID' for i in range(validation)])
            saving_df.to_csv(f'{path_to_save}/{model}/full_responses.csv', index=False)

        msg = f"Querying model: {model}"
        if verbose > 0:
            print(msg)
        else:
            messages.append(msg)

        results = {}
        for prompt in [col for col in prompts.columns if col != 'ID']:

            msg = f"\tQuerying {prompt}"
            if verbose > 0:
                print(msg)
            else:
                messages.append(msg)

            p = {prompt: {}}
            for idx, row in prompts.iterrows():
                current_prompt = row[prompt]
                current_id = row["ID"]

                msg = f"\t\tCase ID: {current_id}"
                if verbose > 1:
                    print(msg)
                else:
                    messages.append(msg)

                case_answers = []
                for i in range(validation):

                    response = ollama.chat(
                        model=model,
                        messages=[
                            {"role": "user", "content": current_prompt}
                        ]
                    )

                    response = response["message"]["content"]
                    response = fix_response(response)

                    if verbose > 3:
                        print("\t\tResponse:", response)
                    update_progress_bar()

                    try:

                        json_response = json.loads(response)

                        result = json_response.get('resposta', '')
                        explanation = json_response.get('explicacao', '')
                        case_answers.append({"answer": result, "explanation": explanation})

                    except json.JSONDecodeError:

                        msg = f"\t\t\tFailed to decode JSON"
                        if verbose > 2:
                            print(msg)
                        else:
                            messages.append(msg)

                        case_answers.append({"answer": "Failed JSON", "explanation": "Failed json"})
                    
                    if path_to_save:   #Salvando as respostas enquanto elas são geradas                 

                        if current_id in saving_df["ID"].values:
                            # find its index
                            idx = saving_df.index[saving_df["ID"] == current_id][0]
                        else:
                            # create a new row
                            idx = len(saving_df)
                            saving_df.loc[idx, "ID"] = current_id

                        saving_df.loc[idx, f'{prompt} ({i+1}x)'] = result
                        saving_df.loc[idx, f'{prompt} ({i+1}x) Explanation'] = explanation

                        saving_df.to_csv(f'{path_to_save}/{model}/full_responses.csv', index=False)
                            

                p[prompt].update({current_id: case_answers})

            results.update(p)
        
        m = modelAnswer(
            model=model,
            validation=validation,
            prompts_used=prompts.shape[1] - 1,
            n_cases=prompts.shape[0],
            responses=results
        )

        responses.append(m)

    return responses