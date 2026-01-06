import json
import os
from datetime import datetime
import pandas as pd

from api.summarize import batch_summary
from tools.config import (
    r1_max_tokens,
    r2_max_tokens,
    r3_max_tokens,
    r4_max_tokens,
    max_prompts_tokens_3R,
)
from tools.llm import LLM
from tools.utils import postprocess_response
from transformers import AutoTokenizer


def calculate_text_tokens(text):
    tokenizer = AutoTokenizer.from_pretrained(
        "./tokenizer", trust_remote_code=False, use_fast=True
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)

    return len(tokens)


class DataStory:
    def __init__(
        self,
        dataset_name,
        data_df: pd.DataFrame,
        write_stages=None,
    ):
        if write_stages is None:
            write_stages = ["reason", "revalidate", "reflect", "final"]
        self.messages = []
        self.result = ""
        self.dataset_name = dataset_name
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

    def calc_3r_tokens(self):
        remain_tokens = (
            self.llm.max_tokens
            - max_prompts_tokens_3R
            - (r1_max_tokens + r2_max_tokens + r3_max_tokens)
        )
        return remain_tokens

    def reason(self, data_samples, data_summary=""):
        """
        Generate a data story from the provided dataset using reasoning llm.
        """
        self.messages += [
            self.reason_system_prompt,
            {
                "role": "user",
                "content": f"Generate a data story in json format for the following dataset:\n{json.dumps(data_samples, ensure_ascii=False)}\n {data_summary} The data story must includes story_title, story_subtitle, and five story_pieces. Each story_piece contains narration (the discovered data facts), question (the question posed about the data fact), and visualization (the chart type that best visualizes the data fact for the question). For example,\n{self.example}\nFor visualization, when the underlying narrations are suitable, prioritize attempting to use complex chart types. For example, bar chart races are suitable for dynamically showing changes over a temporal axis.",
            },
        ]
        start_time = datetime.now()
        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=self.messages,
            stream=False,
            temperature=1.0,
            max_tokens=r1_max_tokens,
            response_format={"type": "json_object"},
        )
        elapsed_time = (datetime.now() - start_time).total_seconds()
        content = completion.choices[0].message.content
        print("reason:\n", content)
        content = postprocess_response(content)
        self.reason_results = content
        self.result = content
        # add response into chat history
        self.messages.append({"role": "assistant", "content": content})
        return {
            "content": content,
            "usage": completion.usage,
            "elapsed_time": elapsed_time,
        }

    def check_data_fact(self, data, data_summary, narration):
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
                "content": f"The data summary is \n{data_summary}\n The data fact is \n{narration}\n",
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
            data_fata_check_result = ns["data_fact_validate"](pd.DataFrame(data))
            return data_fata_check_result
        except Exception as e:
            print(e)
            return {}

    def reflection(self, data_summary, data_fact_check_results=None):
        if data_fact_check_results is None:
            data_fact_check_results = []
        self.messages += [
            {
                "role": "system",
                "content": f"""Based on the given data summary and data fact validation results, reflect on whether the data story above contains any subjective and objective issues. Subjective issues include aspects such as logical consistency, narrative flow, the overall appeal of the story, and whether all visualizations are too uniform in type. Objective issues relate to data accuracy, whether the `visualization` in each story accurately and thoroughly fulfills the requirements outlined in the `question`, whether the referenced columns exist, and other verifiable facts. The data summary includes the data type, data samples, statistical information, and the count of unique values for each column in the dataset. For columns with the data type 'category', it also provides the frequency of each enumerated value. provide accurate calculations generated for the narration of each story piece and can be used to verify whether the narration is accurate. Only return the issues and don't do any revision.""",
            },
            {
                "role": "user",
                "content": f"Do not output any confirmation when the narration is correct. Treat all numeric deviations within ±0.05 (or ±0.1%) as negligible and do not classify them as errors. If it is not possible to determine the correctness of the narration from the provided information, assume it is correct by default and do not flag it as an error. The data summary is \n{data_summary}\n The data fact validation results are \n{data_fact_check_results}\nLet the reflection begin now. ",
            },
        ]
        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=self.messages,
            stream=False,
            temperature=1.0,
            max_tokens=r2_max_tokens,
        )
        content = completion.choices[0].message.content
        print("reflect:\n", content)
        self.reflect_results = content
        # add response into chat history
        self.messages.append({"role": "assistant", "content": content})
        return True if content else False

    def refine(self):
        self.messages.append(
            {
                "role": "user",
                "content": "Fix only clearly identifiable errors in the narration, do not attempt to correct unverifiable objective issues. And provide a new data story in json format. Don't do any explanation.",
            }
        )
        start_time = datetime.now()
        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=self.messages,
            stream=False,
            temperature=1.0,
            max_tokens=r3_max_tokens,
        )
        elapsed_time = (datetime.now() - start_time).total_seconds()
        content = completion.choices[0].message.content
        content = postprocess_response(content)
        print("refine:\n", content)
        self.result = content
        return {
            "content": content,
            "usage": completion.usage,
            "elapsed_time": elapsed_time,
        }

    def revalidate(self, data_samples, data_summary):
        messages = [
            {
                "role": "system",
                "content": """You are provided with an incomplete data story generated from an initial dataset, along with a supplementary dataset and its corresponding data summary. Your task is to update the data story by incorporating facts from the supplementary dataset without directly replacing the original narration. Instead, you should combine the new data with the existing narration. Carefully analyze the supplementary dataset and data summary to verify whether `story_subtitle` and each `narration` in the story needs to be updated. If necessary, modify the narration to reflect the updated facts—this could involve arithmetic operations, comparisons, or other data calculations. For example, if the old narration mentions that category A has the highest count, with a value of 100, and the new data shows that category B has a count of 120, the old narration should be updated to state that category B has the highest count, with a value of 120. Similarly, if the old narration mentions category A with a value of 100, and the new data shows category A with a value of 50, perform the necessary calculation (100 + 50 = 150) and update the narration accordingly. If the narration involves proportions or percentages, calculate the new proportion or percentage from the supplementary data, then add it to the old value to obtain the final result. The goal is to combine the new data with the original narration, ensuring the updated story reflects the facts from the supplementary dataset accurately while preserving the context and integrity of the original narrative.""",
            },
            {
                "role": "user",
                "content": f"The supplementary data consists of:\n{json.dumps(data_samples, ensure_ascii=False)}\n. The corresponding data summary is \n{data_summary}\n. The content of the current data story is: {self.result}. Validate the current data story and return the data story after update. Don't do any explanation and do not show any calculation process in the new data story.",
            },
        ]

        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            stream=False,
            temperature=1.0,
            max_tokens=r4_max_tokens,
        )
        content = completion.choices[0].message.content
        print("revalidate:\n", content)
        self.result = postprocess_response(content)

    def write(self):
        dist = f"./results/{self.dataset_name}"
        exist_count = len(
            list(filter(lambda x: x.startswith("data_story_"), os.listdir(dist)))
        )
        if "final" in self.write_stages:
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
        if "revalidate" in self.write_stages:
            with open(
                f"{dist}/data_story_{exist_count + 1}_revalidate.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(self.data_fact_check_results, default=str, indent=4))
        if "reflect" in self.write_stages:
            with open(
                f"{dist}/data_story_{exist_count + 1}_reflect.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(self.reflect_results)

    def get_datas_by_tokens(self, start_index, remain_tokens):
        """
        Get data samples by tokens.
        """
        data_samples = []
        stop_index = None
        for index, line in self.data.iloc[start_index:].iterrows():
            line = line.to_dict()
            if index < start_index:
                continue
            line_tokens = self.llm.calc_tokens(json.dumps(line, ensure_ascii=False))
            if line_tokens > remain_tokens:
                stop_index = index
                break
            data_samples.append(line)
            remain_tokens -= line_tokens
        if not stop_index:
            stop_index = len(self.data)
        return data_samples, stop_index

    def run_4r(self):
        first_batch_data, stop_index = self.get_datas_by_tokens(
            0, remain_tokens=self.calc_3r_tokens()
        )
        first_batch_summary = batch_summary(first_batch_data)
        print(
            f"dataset_name:{self.dataset_name}, module: data story generation, phase:4R start"
        )
        self.reason(first_batch_data)
        data_fact_check_results = []
        for story_piece in json.loads(self.reason_results)["story_pieces"]:
            check_result = self.check_data_fact(
                first_batch_data, first_batch_summary, story_piece["narration"]
            )
            data_fact_check_results.append(check_result)
        print("data fact check results: ", data_fact_check_results)
        self.data_fact_check_results = data_fact_check_results
        reflection_not_null = self.reflection(
            first_batch_summary, data_fact_check_results
        )
        if reflection_not_null:
            self.refine()
        print("batch end index: ", stop_index)
        print(
            f"dataset_name:{self.dataset_name}, module: data story generation, phase:4R finish"
        )
        self.write()
        return self.result

    def run_3r_1r(self):
        first_batch_data, stop_index = self.get_datas_by_tokens(
            0, remain_tokens=self.calc_3r_tokens()
        )
        first_batch_summary = batch_summary(first_batch_data)
        print(
            f"dataset_name:{self.dataset_name}, module: data story generation, phase:3R start"
        )
        self.reason(first_batch_data)
        self.reflection(first_batch_summary)
        self.refine()
        print("batch end index: ", stop_index)
        print(
            f"dataset_name:{self.dataset_name}, module: data story generation, phase:3R finish"
        )
        if stop_index < len(self.data):
            print(
                f"dataset_name:{self.dataset_name}, module: data story generation, phase:1R start"
            )
            while stop_index < len(self.data):
                validation_batch_data, stop_index = self.get_datas_by_tokens(
                    stop_index, remain_tokens=self.llm.max_tokens - r4_max_tokens
                )
                print("batch end index: ", stop_index)
                validation_batch_summary = batch_summary(validation_batch_data)
                self.revalidate(validation_batch_data, validation_batch_summary)
            print(
                f"dataset_name:{self.dataset_name}, module: data story generation, phase:1R finish"
            )
        self.write()
        return self.result

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
