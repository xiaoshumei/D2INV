import os
import json

from api.data_story import DataStory
from api.summarize import file_summary
from dotenv import load_dotenv


data_files = [
    "barley.json",
    "burtin.json",
    "co2-concentration.csv",
    "disasters.csv",
    "driving.json",
    "gapminder-health-income.csv",
    "github.csv",
    "iowa-electricity.csv",
    "la-riots.csv",
    "ohlc.json",
]


def reason_without_summary(index):
    for data_file in data_files:
        dataset_name = data_file.split(".")[0]
        [data_summary, data] = file_summary(data_file)
        data_story = DataStory(dataset_name, data, data_summary, write_stages=[])
        story = data_story.reason()
        data_fact_check_results = []
        for story_piece in json.loads(story)["story_pieces"]:
            check_result = data_story.check_data_fact(story_piece["narration"])
            data_fact_check_results.append(check_result)
        dist_dir = f"./experiments/data_story/reason_without_summary/{dataset_name}"
        os.makedirs(
            dist_dir,
            exist_ok=True,
        )
        with open(
            f"{dist_dir}/reason_{index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(story)
        with open(
            f"{dist_dir}/review_{index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(data_fact_check_results, default=str, indent=4))


def reason_with_summary(index):
    for data_file in data_files:
        dataset_name = data_file.split(".")[0]
        [data_summary, data] = file_summary(data_file)
        data_story = DataStory(dataset_name, data, data_summary, write_stages=[])
        story = data_story.reason(data_summary=data_summary)
        data_fact_check_results = []
        for story_piece in json.loads(story)["story_pieces"]:
            check_result = data_story.check_data_fact(story_piece["narration"])
            data_fact_check_results.append(check_result)
        dist_dir = f"./experiments/data_story/reason_with_summary/{dataset_name}"
        os.makedirs(
            dist_dir,
            exist_ok=True,
        )
        with open(
            f"{dist_dir}/reason_{index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(story)
        print(data_fact_check_results)
        with open(
            f"{dist_dir}/review_{index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(data_fact_check_results, default=str, indent=4))


def reflect_with_only_summary(index):
    for data_file in data_files:
        dataset_name = data_file.split(".")[0]
        [data_summary, data] = file_summary(data_file)
        data_story = DataStory(dataset_name, data, data_summary, write_stages=[])
        with open(
            f"experiments/data_story/reason_with_summary/{dataset_name}/reason_{index}.json",
            "r",
            encoding="utf-8",
        ) as f:
            story = json.load(f)
            data_story.reason_results = story
        data_fact_check_results = []
        reflection_not_null = data_story.reflection()
        if reflection_not_null:
            story = data_story.refine()
        dist_dir = f"./experiments/data_story/reflect_with_only_summary/{dataset_name}"
        os.makedirs(
            dist_dir,
            exist_ok=True,
        )
        with open(
            f"{dist_dir}/reason_{index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(story)
        for story_piece in json.loads(story)["story_pieces"]:
            check_result = data_story.check_data_fact(story_piece["narration"])
            data_fact_check_results.append(check_result)
        with open(
            f"{dist_dir}/review_{index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(data_fact_check_results, default=str, indent=4))


def reflect_with_summary_and_review(index):
    for data_file in data_files:
        dataset_name = data_file.split(".")[0]
        [data_summary, data] = file_summary(data_file)
        data_story = DataStory(dataset_name, data, data_summary, write_stages=[])
        with open(
            f"experiments/data_story/reason_with_summary/{dataset_name}/reason_{index}.json",
            "r",
            encoding="utf-8",
        ) as f:
            story = json.load(f)
            data_story.reason_results = story
        data_fact_check_results = []
        for story_piece in story["story_pieces"]:
            check_result = data_story.check_data_fact(story_piece["narration"])
            data_fact_check_results.append(check_result)
        reflection_not_null = data_story.reflection(data_fact_check_results)
        if reflection_not_null:
            story = data_story.refine()
        dist_dir = (
            f"./experiments/data_story/reflect_with_summary_and_review/{dataset_name}"
        )
        os.makedirs(
            dist_dir,
            exist_ok=True,
        )
        with open(
            f"{dist_dir}/reason_{index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(story)
        # check the final data facts
        check_results = []
        for story_piece in json.loads(story)["story_pieces"]:
            check_result = data_story.check_data_fact(story_piece["narration"])
            check_results.append(check_result)
        with open(
            f"{dist_dir}/review_{index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(check_results, default=str, indent=4))


if __name__ == "__main__":
    load_dotenv()
    for i in range(9, 11):
        reason_without_summary(i)
        reason_with_summary(i)
        reflect_with_only_summary(i)
        reflect_with_summary_and_review(i)
