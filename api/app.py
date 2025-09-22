import os
import json
from dotenv import load_dotenv
import argparse
from pathlib import Path
import sys


# 自动检测并设置项目路径
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


# generate an inv for every dataset
def experiments(dataset_dir):
    # get data filename from dataset_dir
    data_file = os.path.basename(dataset_dir)
    dataset_name = data_file.split(".")[0]
    os.makedirs(f"./results/{dataset_name}", exist_ok=True)
    print(f"dataset_name:{dataset_name} start to summarize")
    [data_summary, data] = file_summary(data_file)
    write_summary(dataset_name, data_summary)
    print(f"dataset_name:{dataset_name}, start to generate data story")
    data_story = DataStory(dataset_name, data).run()
    print(f"dataset_name:{dataset_name}, start to generate visualization template")
    html_template = InfographicTemplate(dataset_name, data_story).run()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, help="Dataset path")
    args = parser.parse_args()

    if args.datasets:
        dataset_path = args.datasets
        print(f"Dataset path: {dataset_path}")
        load_dotenv()
        experiments(dataset_path)
    else:
        print("Dataset path parameter not provided")
