import json
import os
import pandas as pd

from tools.llm import LLM
from tools.utils import postprocess_response, fix_json


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

    def _schema_ok(self, parsed) -> bool:
        """True if the parsed story is in the required JSON schema."""
        if not isinstance(parsed, dict):
            return False
        pieces = parsed.get("story_pieces")
        if not isinstance(pieces, list) or not pieces:
            return False
        return all(
            isinstance(p, dict)
            and p.get("narration")
            and p.get("question")
            and p.get("visualization")
            for p in pieces
        )

    def reason(self, data_summary=None, force_schema_hint: str = ""):
        """
        Generate a data story from the provided dataset using reasoning llm.

        When force_schema_hint is provided (after a schema violation), the model
        is asked again to return exactly the required JSON structure.
        """
        template = json.dumps(
            {
                "story_title": "string",
                "story_subtitle": "string",
                "story_pieces": [
                    {
                        "narration": "string",
                        "question": "string",
                        "visualization": "string",
                    }
                ],
            },
            indent=2,
        )
        # Compact dataset preview: don't inline the full dataset into the prompt
        # (large payloads can stall large models or hit request limits).
        df = self.data
        preview = (
            {"num_rows": int(len(df)), "columns": list(df.columns)[:50]}
            if df is not None
            else {}
        )
        head_json = (
            df.head(8).to_dict(orient="records") if df is not None else []
        )
        user_content = (
            f"Generate a data story for the following dataset as a JSON object.\n"
            f"Dataset overview:\n{json.dumps(preview, ensure_ascii=False)}\n"
            f"First rows:\n{json.dumps(head_json, ensure_ascii=False, default=str)}\n"
            f"Dataset summary:\n{json.dumps(data_summary, ensure_ascii=False) if data_summary else ''}\n\n"
            f"The JSON MUST contain exactly these fields: story_title (string), "
            f"story_subtitle (string), and story_pieces, an array of exactly 5 objects, "
            f"each with narration (string), question (string), visualization (string).\n"
            f"Use a clear, data-driven narrative. For visualization, prefer complex chart "
            f"types (e.g. bar chart race) when the underlying narration fits.\n\n"
            f"Required JSON shape (fill in values, keep this exact structure):\n{template}\n\n"
            f"Return ONLY the JSON object. No code fences, no extra text, no message fields "
            f"such as role/content/messages, no 'authors' field."
        )
        if force_schema_hint:
            user_content += (
                "\n\nYour previous answer did NOT follow the required JSON schema. "
                "Return a JSON object with EXACTLY these keys: "
                "\"story_title\" (string), \"story_subtitle\" (string), \"story_pieces\" "
                "(array of exactly 5 objects, each with string keys "
                "\"narration\", \"question\", \"visualization\"). "
                "Do NOT return a freeform 'story'/'sections' text field. Return only valid JSON."
            )
        messages = [
            self.reason_system_prompt,
            {
                "role": "user",
                "content": user_content,
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
        print("reason:\n", content)
        # Robustly extract the first valid JSON object from whatever the model returned.
        from tools.utils import extract_json_object

        parsed = extract_json_object(content)
        if parsed is None:
            # Fall back to the original string so downstream code can still see it.
            self.reason_results = postprocess_response(content)
        else:
            self.reason_results = json.dumps(parsed, ensure_ascii=False)
        self.result = self.reason_results
        return self.reason_results

    def check_data_fact(self, narration):
        try_max = 3
        code_scaffold = """
        def data_fact_validate(df):
            # calculations ...

            results = {
                ...
            }

            return results"""
        system_prompt = f"Given a narration and a data summary, generate Python + pandas code to compute and validate the data facts based on a preloaded df. Just fill in the following code template:\n{code_scaffold}, don't do any explanation."
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
        while try_max:
            try:
                completion = self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=messages,
                    stream=False,
                    temperature=1.0,
                )
                data_fata_check_code = completion.choices[0].message.content
                content = postprocess_response(data_fata_check_code)
                ns = {"pd": pd}
                exec(content, ns)
                data_fata_check_result = ns["data_fact_validate"](self.data)
                fixed_result = {}
                for key in data_fata_check_result.keys():
                    # sometimes, key will be a dict
                    if isinstance(key, tuple):
                        fixed_result["_".join(str(x) for x in key)] = (
                            data_fata_check_result[key]
                        )
                    else:
                        fixed_result[key] = data_fata_check_result[key]
                return fixed_result
            except Exception as e:
                print(
                    "Error on generating review code, remaining retry times: ",
                    try_max - 1,
                    "Error is: ",
                    e,
                )
                try_max -= 1
        print("Fail to generate review code for current story piece")
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
        content = fix_json(content)
        print("refine:\n", content)
        self.result = content
        return content

    def _parse_normalized_story(self, raw):
        """Vend a well-formed story dict even when the LLM returns a freeform
        narrative (e.g. `{"story": "..."}`) instead of the required
        `story_pieces` schema. This prevents `KeyError: 'story_pieces'` from
        crashing the 4R pipeline."""
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}

        if not isinstance(parsed, dict):
            parsed = {}

        # Already well-formed: has a non-empty story_pieces list.
        pieces = parsed.get("story_pieces")
        if isinstance(pieces, list) and pieces:
            return parsed

        # Freeform narrative: upgrade it into a single story piece so the rest
        # of the pipeline (data-fact check, reflection, refine, write) can run.
        if "story" in parsed and isinstance(parsed["story"], str) and parsed["story"].strip():
            narrative = parsed["story"].strip()
            return {
                "story_title": parsed.get("story_title", "Data Story"),
                "story_subtitle": parsed.get("story_subtitle", ""),
                "story_pieces": [
                    {
                        "narration": narrative,
                        "question": "What does this dataset reveal?",
                        "visualization": "bar chart",
                    }
                ],
            }

        # Last resort: keep the original payload so nothing crashes.
        return parsed

    def write(self):
        # dataset_name may carry a file extension; always derive the stem so
        # we write into ./results/<name> (matching api/app.py and the other
        # downstream modules). Files use stable names so existing results can
        # be re-read and served without regenerating.
        dataset_stem = os.path.splitext(self.dataset_name)[0]
        dist = f"./results/{dataset_stem}"
        os.makedirs(dist, exist_ok=True)
        if "refine" in self.write_stages:
            with open(
                f"{dist}/data_story.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(self.result)
        if "reason" in self.write_stages:
            with open(
                f"{dist}/data_story_reason.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(self.reason_results)
        if "reflect" in self.write_stages:
            with open(
                f"{dist}/data_story_reflect.json",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(self.data_fact_check_results, indent=4))

    def run_4r(self):
        print(
            f"dataset_name:{self.dataset_name}, module: data story generation, phase:4R start"
        )
        # Generate the story, but if the model returns a freeform narrative
        # (e.g. {"story": ...}) instead of the required structured schema,
        # ask once more with an explicit corrective hint.
        self.reason(self.data_summary)
        parsed = self._parse_normalized_story(self.reason_results)
        if not self._schema_ok(parsed):
            print("reason output missing required story_pieces schema; re-asking once")
            self.reason(self.data_summary, force_schema_hint=True)
            parsed = self._parse_normalized_story(self.reason_results)
        data_fact_check_results = []
        for story_piece in parsed.get("story_pieces", []):
            check_result = self.check_data_fact(story_piece.get("narration", ""))
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
        return self._parse_normalized_story(self.result)

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
