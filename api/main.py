import os
import json
from dotenv import load_dotenv
import sys
from pathlib import Path


def setup_path():
    """Set project root directory to Python path"""
    current_file = Path(__file__).resolve()

    project_root = current_file.parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added project path: {project_root}")


# Set python path
setup_path()
from data_story import DataStory
from summarize import file_summary, write_summary
from infographic_template import InfographicTemplate
from inv import INV
from evaluate import Evaluate


# get task list
def get_data_files():
    data_dir = "./datasets/"
    data_list = os.listdir(data_dir)
    return data_list


# generate an inv for every dataset
def experiments():
    data_list = get_data_files()
    for data_file in data_list:
        if data_file != "stocks.csv":
            continue
        print(f"data_file:{data_file} start")
        dataset_name = data_file.split(".")[0]
        os.makedirs(f"./results/{dataset_name}", exist_ok=True)
        print(f"dataset_name:{dataset_name} start to summarize")
        [data_summary, data] = file_summary(data_file)
        write_summary(dataset_name, data_summary)
        print(f"dataset_name:{dataset_name}, start to generate data story")
        # data_story = DataStory(dataset_name, data).run()
        with open(
            f"./results/{dataset_name}/data_story_1.json", "r", encoding="utf-8"
        ) as f:
            data_story = json.dumps(json.load(f))
        print(f"dataset_name:{dataset_name}, start to generate visualization template")
        html_template = InfographicTemplate(dataset_name, data_story).run()
        # with open(
        #     f"./results/{dataset_name}/infographic_template_1.html",
        #     "r",
        #     encoding="utf-8",
        # ) as f:
        #     html_template = f.read()
        print(
            f"dataset_name:{dataset_name}, start to automatically generate data visualization and inv"
        )
        inv_no_data = INV(
            dataset_name,
            json.loads(data_story),
            data_summary,
            html_template,
            data,
        ).run()
        print(f"dataset_name:{dataset_name}, start to evaluate inv")
        Evaluate(dataset_name, inv_no_data).run()
        print(f"data_file:{data_file} end")


if __name__ == "__main__":
    load_dotenv()
    experiments()
