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
                 model: str = None, 
                 verbose: int = 0,
                 ) -> list[modelAnswer]:

    if model is None or model == "Todos":
        models = get_ollama_models()
    else:
        models = [model]

    total_queries = len(models) * (prompts.shape[1] - 1) * prompts.shape[0] * validation
    current_query = 0

    def update_progress_bar():
        nonlocal current_query
        current_query += 1
        progress = current_query / total_queries

        if hasattr(query_models, "progress_callback") and callable(query_models.progress_callback):
            query_models.progress_callback(progress, current_query, total_queries)

    responses = []
    for model in models:
        if verbose > 0:
            print("Querying model:", model)

        results = {}
        for prompt in [col for col in prompts.columns if col != 'ID']:
            if verbose:
                print("\tQuerying ", prompt)
            p = {prompt: {}}
            for idx, row in prompts.iterrows():
                current_prompt = row[prompt]
                current_id = row["ID"]
                if verbose > 1:
                    print("\t\tQuerying case ", current_id)
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
                        if verbose > 2:
                            print(json_response)
                        result = json_response.get('resposta', '')
                        explanation = json_response.get('explicacao', '')
                        case_answers.append({"answer": result, "explanation": explanation})
                    except json.JSONDecodeError:
                        if verbose > 2:
                            print("\t\t\tFailed to decode JSON")
                        case_answers.append({"answer": "Failed JSON", "explanation": "Failed json"})
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