import json
import pandas as pd

class modelAnswer:
    model: str
    validation: int
    prompts_used: int
    n_cases: int
    responses: dict

    def __init__(self, model, validation, prompts_used, n_cases, responses):
        self.model = model
        self.validation = validation
        self.prompts_used = prompts_used
        self.n_cases = n_cases
        self.responses = responses

    def print_summary(self):
        print(f"Model: {self.model}")
        print(f"Prompts Used: {self.prompts_used}")
        print(f"Number of Cases: {self.n_cases} | Validation: {self.validation}")

    def saveJson(self):
        with open(f"{self.model}_responses.json", "w", encoding="utf-8") as f:
            json.dump({"model": self.model, 
                        "validation": self.validation,
                        "prompts_used": self.prompts_used,
                        "n_cases": self.n_cases,
                        "responses": json.loads(json.dumps(self.responses))}, f, indent=4)
            
    def display(self):
        print(f"Model: {self.model}")
        print(f"Validation: {self.validation}")
        print(f"Prompts Used: {self.prompts_used}")
        print(f"Number of Cases: {self.n_cases}")
        print("Responses:")
        for prompt, responses in self.responses.items():
            print(f"  {prompt}:")
            for case_id, case_responses in responses.items():
                print(f"    Case {case_id}:")
                for resp in case_responses:
                    print(f"      Answer: {resp['answer']}")
                    print(f"      Explanation: {resp['explanation']}")

    def build_responses_df(self):   

        dfs = []

        ids = []
        first_prompt = next(iter(self.responses.values()))
        for case_id in first_prompt.keys():
            ids.append(case_id)

        dfs.append(pd.DataFrame(data=ids, columns=['ID']))

        for prompt, cases in self.responses.items():
            
            data = []
            columns = [f'{prompt} ({i+1}x)' for i in range(self.validation)]
            
            for case, validations in cases.items():
                #print(case, type(validations))
                current_case_data = []
                for test in validations:
                    current_case_data.append(test["answer"])
                    #print(current_case_data)
                data.append(current_case_data)
            dfs.append(pd.DataFrame(data=data, columns=columns))
        return pd.concat(dfs, ignore_index=False, axis=1)

    def get_case_responses(self, correct_answers: pd.DataFrame = None):
        results = self.build_responses_df()
        ids = results['ID']
        results.drop(columns=['ID'], inplace=True)
        prompt_results =[]

        for i in range(self.prompts_used):
            prompt_answers = results.iloc[:, i*self.validation:i*self.validation+self.validation]
            prompt_name = prompt_answers.columns[0][:-4]
            #prompt_answers[f"{prompt_name}(moda)"] = prompt_answers.mode(axis=1)[0]
            prompt_answers.loc[:, f"{prompt_name}(moda)"] = prompt_answers.mode(axis=1)[0]
            ids = pd.concat([ids, prompt_answers[f"{prompt_name}(moda)"]], axis=1)
            results_mode= pd.concat([ids["ID"], prompt_answers], axis=1)

            if correct_answers is not None:
                results_mode = pd.merge(results_mode, correct_answers, on="ID", how="left")

            prompt_results.append(results_mode)
        return ids, prompt_results
    
    def get_cases_responses_with_mode(self):

        a = self.build_responses_df()
        b, c = self.get_case_responses()
        return pd.concat([a.iloc[:, 0], b.iloc[:, 1:], a.iloc[:, 1:]], axis=1)
    
    def get_an_explanation(self, prompt, case_id, answer):
        for resp in self.responses[prompt][case_id]:
            if resp['answer'] == answer:
                return resp['explanation']