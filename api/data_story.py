import json
import os
from datetime import datetime
import pandas as pd

from tools.llm import LLM
from tools.utils import postprocess_response


class DataStory:
    def __init__(
        self,
        dataset_name,
        data_df: pd.DataFrame,
        data_summary,
        write_stages=None,
    ):
        if write_stages is None:
            write_stages = ["refine"]
        self.result = ""
        self.dataset_name = dataset_name
        self.data_summary = data_summary
        self.data = data_df
        self.llm = LLM()
        self.write_stages = write_stages
        self.data_fact_check_results = []
        self.reason_results = ""
        self.reflect_results = ""
        self.example = {
            "story_title": "COVID-19: A Global Pandemic in Numbers",
            "story_subtitle": "A Comprehensive Analysis of Infection Rates, Vaccination Trends, and Mortality Patterns Across Countries",
            "story_pieces": [
                {
                    "narration": "The United States (over 80 million cases), India (over 45 million cases), and Brazil (over 30 million cases) lead in total cases. 50% of global COVID-19 cases are concentrated in just 5 countries.",
                    "question": "Which countries have the highest total number of COVID-19 cases, and how is the distribution skewed?",
                    "visualization": "ranked bar chart (countries vs. total cases)",
                },
                {
                    "narration": "Countries with the highest vaccination rates include the United Arab Emirates (over 95% vaccinated), Portugal (over 85%), and Chile (over 80%). 70% of the global population has received at least one dose of the vaccine.",
                    "question": "Which countries have the highest vaccination rates, and what is the global vaccination coverage?",
                    "visualization": "world map (countries colored by vaccination rate)",
                },
                {
                    "narration": "The highest COVID-19 mortality rates are observed in countries such as Brazil, Mexico, and Russia, with mortality rates exceeding 3%. In contrast, countries like Australia and New Zealand have kept mortality rates below 1%.",
                    "question": "Which countries are experiencing the highest mortality rates from COVID-19?",
                    "visualization": "bubble chart (countries vs. mortality rate)",
                },
            ],
        }

        self.reason_system_prompt = {
            "role": "system",
            "content": "You are a data analyst who excels at mining data insights and data stories from datasets. When given a dataset, you just return a json and don't explain anything. Do not include any comments or additional text in json.",
        }

    def reason(self, data_summary=None):
        """
        Generate a data story from the provided dataset using reasoning llm.
        """
        messages = [
            self.reason_system_prompt,
            {
                "role": "user",
                "content": f"Generate a data story in json format for the following dataset:\n{json.dumps(self.data.to_dict(orient='records'), ensure_ascii=False)}\n A summary of the dataset is \n{json.dumps(data_summary,ensure_ascii=False) if data_summary else ''}\n The data story must includes story_title, story_subtitle, and five story_pieces. Each story_piece contains narration (the discovered data facts), question (the question posed about the data fact), and visualization (the chart type that best visualizes the data fact for the question). For example,\n{self.example}\nFor visualization, when the underlying narrations are suitable, prioritize attempting to use complex chart types. For example, bar chart races are suitable for dynamically showing changes over a temporal axis.",
            },
        ]
        start_time = datetime.now()
        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            stream=False,
            temperature=1.0,
            response_format={"type": "json_object"},
        )
        elapsed_time = (datetime.now() - start_time).total_seconds()
        content = completion.choices[0].message.content
        print("reason:\n", content)
        content = postprocess_response(content)
        self.reason_results = content
        self.result = content
        return {
            "content": content,
            "usage": completion.usage,
            "elapsed_time": elapsed_time,
        }

    def check_data_fact(self, narration):
        code_scaffold = """
        def data_fact_validate(df):
            # calculations ...

            results = {
                ...
            }

            return results"""
        system_prompt = f"Given a narration and a data summary, generate Python + pandas code to compute and validate the data facts based on a preloaded df. Just fill in the following code template:\n{code_scaffold}, don't do any explanation"
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"The data summary is \n{self.data_summary}\n The data fact is \n{narration}\n",
            },
        ]

        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            stream=False,
            temperature=1.0,
        )
        data_fata_check_code = completion.choices[0].message.content
        content = postprocess_response(data_fata_check_code)
        ns = {"pd": pd}
        try:
            exec(content, ns)
            data_fata_check_result = ns["data_fact_validate"](self.data)
            return data_fata_check_result
        except Exception as e:
            print(e)
            return {}

    def reflection(self, data_fact_check_results=None):
        if data_fact_check_results is None:
            data_fact_check_results = []
        messages = [
            {
                "role": "system",
                "content": f"""Based on the given data summary and data fact validation results, reflect on whether a data story contains any subjective and objective issues. Subjective issues include aspects such as logical consistency, narrative flow, the overall appeal of the story, and whether all visualizations are too uniform in type. Objective issues relate to data accuracy, whether the `visualization` in each story accurately and thoroughly fulfills the requirements outlined in the `question`, whether the referenced columns exist, and other verifiable facts. The data summary includes the data type, data samples, statistical information, and the count of unique values for each column in the dataset. For columns with the data type 'category', it also provides the frequency of each enumerated value. provide accurate calculations generated for the narration of each story piece and can be used to verify whether the narration is accurate. Only return the issues and don't do any revision.""",
            },
            {
                "role": "user",
                "content": f"Do not output any confirmation when the narration is correct. Treat all numeric deviations within ±0.05 (or ±0.1%) as negligible and do not classify them as errors. If it is not possible to determine the correctness of the narration from the provided information, assume it is correct by default and do not flag it as an error. The data story is \n{self.reason_results}\n The data summary is \n{self.data_summary}\n The data fact validation results are \n{data_fact_check_results}\n Let the reflection begin now. ",
            },
        ]
        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            stream=False,
            temperature=0.6,
        )
        content = completion.choices[0].message.content
        print("reflect:\n", content)
        self.reflect_results = content

        return True if content else False

    def refine(self):
        messages = [
            {
                "role": "system",
                "content": f"Given an old data story, as well as the reflection results on it, including errors and points to be optimized. Fix the errors and optimize the points, and return a new JSON-formatted data story. Fix only clearly identifiable errors in the narration, do not attempt to correct unverifiable objective issues. Don't do any explanation.",
            },
            {
                "role": "user",
                "content": f"The old data story is \n{self.reason_results}\n The reflection results are \n{self.reflect_results}\n",
            },
        ]
        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            stream=False,
            temperature=0.6,
        )
        content = completion.choices[0].message.content
        content = postprocess_response(content)
        print("refine:\n", content)
        self.result = content

    def write(self):
        dist = f"./results/{self.dataset_name}"
        exist_count = len(
            list(filter(lambda x: x.startswith("data_story_"), os.listdir(dist)))
        )
        if "refine" in self.write_stages:
            with open(
                f"{dist}/data_story_{exist_count + 1}.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(self.result)
        if "reason" in self.write_stages:
            with open(
                f"{dist}/data_story_{exist_count + 1}_reason.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(self.reason_results)
        if "reflect" in self.write_stages:
            with open(
                f"{dist}/data_story_{exist_count + 1}_reflect.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(self.data_fact_check_results, indent=4))

    def run_4r(self):
        print(
            f"dataset_name:{self.dataset_name}, module: data story generation, phase:4R start"
        )
        self.reason(self.data_summary)
        data_fact_check_results = []
        for story_piece in json.loads(self.reason_results)["story_pieces"]:
            check_result = self.check_data_fact(story_piece["narration"])
            data_fact_check_results.append(check_result)
        print("data fact check results: ", data_fact_check_results)
        self.data_fact_check_results = data_fact_check_results
        reflection_not_null = self.reflection(data_fact_check_results)
        if reflection_not_null:
            self.refine()
        print(
            f"dataset_name:{self.dataset_name}, module: data story generation, phase:4R finish"
        )
        self.write()
        return json.loads(self.result)

    def edit(self, prompts):
        """
        Modify self.result based on user input prompts
        """
        messages = [
            {
                "role": "system",
                "content": """You are a data story editor. Your task is to modify the existing data story based on user prompts while maintaining the JSON format. The data story consists of story_title, story_subtitle, and story_pieces. Each story_piece contains narration, question, and visualization. Make sure to preserve the overall structure and return valid JSON.""",
            },
            {
                "role": "user",
                "content": f"The current data story is: {self.result}. The user requests the following modifications: {prompts}. Please update the data story accordingly and return the modified version in valid JSON format. Do not include any explanations or comments in the JSON.",
            },
        ]

        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            stream=False,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content

        content = postprocess_response(content)
        print("edit:\n", content)
        self.result = content
        return self.result
