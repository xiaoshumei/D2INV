import os

import pandas as pd
from api.data_story import DataStory
from api.summarize import file_summary
from dotenv import load_dotenv
import json


def reason_without_summary():
    data_file = "cars.json"
    dataset_name = data_file.split(".")[0]

    [data_summary, data] = file_summary(data_file)
    end_index_list = [10, 20, 40, 80, 100, 200, 400, 800, 1600]
    for end_index in end_index_list:
        data_story = DataStory(dataset_name, data, write_stages=[])
        story = data_story.reason(
            data[0:end_index].to_dict(),
        )
        data_fact_check_results = []
        for story_piece in json.loads(story["content"])["story_pieces"]:
            check_result = data_story.check_data_fact(
                data, data_summary, story_piece["narration"]
            )
            data_fact_check_results.append(check_result)
        os.makedirs(
            f"./experiments/data_story/reason_without_summary/{dataset_name}",
            exist_ok=True,
        )
        with open(
            f"./experiments/data_story/reason_without_summary/{dataset_name}/1_{end_index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(story["content"])
        with open(
            f"./experiments/data_story/reason_without_summary/{dataset_name}/1_{end_index}_revalidate.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(data_fact_check_results, default=str, indent=4))
        pd.DataFrame(
            [
                {
                    "name": f"{dataset_name}_1_{end_index}",
                    "prompt_tokens": story["usage"].prompt_tokens,
                    "completion_tokens": story["usage"].completion_tokens,
                    "total_tokens": story["usage"].total_tokens,
                    "elapsed_time": story["elapsed_time"],
                }
            ]
        ).to_json(
            "./experiments/data_story/reason_without_summary/logs.jsonl",
            lines=True,
            orient="records",
            mode="a",
        )


def reason_with_summary():
    data_file = "cars.json"
    dataset_name = data_file.split(".")[0]

    [data_summary, data] = file_summary(data_file)
    end_index_list = [10, 20, 40, 80, 100, 200, 400, 800, 1600]
    for end_index in end_index_list:
        data_story = DataStory(dataset_name, data, write_stages=[])
        story = data_story.reason(
            data[0:end_index].to_dict(), data_summary=data_summary
        )
        data_fact_check_results = []
        for story_piece in json.loads(story)["story_pieces"]:
            check_result = data_story.check_data_fact(
                data, data_summary, story_piece["narration"]
            )
            data_fact_check_results.append(check_result)
        os.makedirs(
            f"./experiments/data_story/reason_with_summary/{dataset_name}",
            exist_ok=True,
        )
        with open(
            f"./experiments/data_story/reason_with_summary/{dataset_name}/1_{end_index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(story["content"])
        with open(
            f"./experiments/data_story/reason_with_summary/{dataset_name}/1_{end_index}_revalidate.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(data_fact_check_results, default=str, indent=4))
        pd.DataFrame(
            [
                {
                    "name": f"{dataset_name}_1_{end_index}",
                    "prompt_tokens": story["usage"].prompt_tokens,
                    "completion_tokens": story["usage"].completion_tokens,
                    "total_tokens": story["usage"].total_tokens,
                    "elapsed_time": story["elapsed_time"],
                }
            ]
        ).to_json(
            "./experiments/data_story/reason_with_summary/logs.jsonl",
            lines=True,
            orient="records",
            mode="a",
        )


def reflect_with_summary():
    data_file = "cars.json"
    dataset_name = data_file.split(".")[0]

    [data_summary, data] = file_summary(data_file)
    end_index_list = [10, 20, 40, 80, 100, 200, 400, 800, 1600]
    for end_index in end_index_list:
        data_story = DataStory(dataset_name, data, write_stages=[])
        story = data_story.reason(data[0:end_index].to_dict())
        data_fact_check_results = []
        for story_piece in json.loads(story)["story_pieces"]:
            check_result = data_story.check_data_fact(
                data, data_summary, story_piece["narration"]
            )
            data_fact_check_results.append(check_result)
        print("data fact check results: ", data_fact_check_results)
        reflection_not_null = data_story.reflection(data_summary)
        if reflection_not_null:
            story = data_story.refine()
        os.makedirs(
            f"./experiments/data_story/reflect_with_summary/{dataset_name}",
            exist_ok=True,
        )
        with open(
            f"./experiments/data_story/reflect_with_summary/{dataset_name}/1_{end_index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(story["content"])
        with open(
            f"./experiments/data_story/reflect_with_summary/{dataset_name}/1_{end_index}_revalidate.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(data_fact_check_results, default=str, indent=4))
        pd.DataFrame(
            [
                {
                    "name": f"{dataset_name}_1_{end_index}",
                    "prompt_tokens": story["usage"].prompt_tokens,
                    "completion_tokens": story["usage"].completion_tokens,
                    "total_tokens": story["usage"].total_tokens,
                    "elapsed_time": story["elapsed_time"],
                }
            ]
        ).to_json(
            "./experiments/data_story/reflect_with_summary/logs.jsonl",
            lines=True,
            orient="records",
            mode="a",
        )


def reflect_with_summary_and_revalidate():
    data_file = "cars.json"
    dataset_name = data_file.split(".")[0]

    [data_summary, data] = file_summary(data_file)
    end_index_list = [10, 20, 40, 80, 100, 200, 400, 800, 1600]
    for end_index in end_index_list:
        data_story = DataStory(dataset_name, data, write_stages=[])
        story = data_story.reason(data[0:end_index].to_dict())
        data_fact_check_results = []
        for story_piece in json.loads(story)["story_pieces"]:
            check_result = data_story.check_data_fact(
                data, data_summary, story_piece["narration"]
            )
            data_fact_check_results.append(check_result)
        print("data fact check results: ", data_fact_check_results)
        reflection_not_null = data_story.reflection(
            data_summary, data_fact_check_results
        )
        if reflection_not_null:
            story = data_story.refine()
        os.makedirs(
            f"./experiments/data_story/reflect_with_summary/{dataset_name}",
            exist_ok=True,
        )
        with open(
            f"./experiments/data_story/reflect_with_summary/{dataset_name}/1_{end_index}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(story["content"])
        with open(
            f"./experiments/data_story/reflect_with_summary/{dataset_name}/1_{end_index}_revalidate.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(data_fact_check_results, default=str, indent=4))
        pd.DataFrame(
            [
                {
                    "name": f"{dataset_name}_1_{end_index}",
                    "prompt_tokens": story["usage"].prompt_tokens,
                    "completion_tokens": story["usage"].completion_tokens,
                    "total_tokens": story["usage"].total_tokens,
                    "elapsed_time": story["elapsed_time"],
                }
            ]
        ).to_json(
            "./experiments/data_story/reflect_with_summary/logs.jsonl",
            lines=True,
            orient="records",
            mode="a",
        )


if __name__ == "__main__":
    load_dotenv()
    reason_without_summary()
    # reason_with_summary()
    # reflect_with_summary()
    # reflect_with_summary_and_revalidate()
