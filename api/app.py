import os
import json
from dotenv import load_dotenv
import argparse


from api.data_story import DataStory
from api.summarize import file_summary, write_summary
from api.infographic_template import InfographicTemplate
from api.inv import INV
from api.evaluate import Evaluate


# generate an inv for every dataset
def d2inv(dataset_dir):
    # get data filename from dataset_dir
    data_file = os.path.basename(dataset_dir)
    dataset_name = data_file.split(".")[0]
    os.makedirs(f"./results/{dataset_name}", exist_ok=True)
    print(f"dataset_name:{dataset_name} start to summarize")
    [data_summary, data] = file_summary(data_file)
    write_summary(dataset_name, data_summary)
    yield {
        "stage": "data_summary",
        "dataset_name": dataset_name,
        "data_summary": data_summary,
        "message": "Data summary generated successfully",
    }
    print(f"dataset_name:{dataset_name}, start to generate data story")
    data_story = DataStory(dataset_name, data).run_4r()
    # with open(
    #     f"./results/{dataset_name}/data_story_1.json", "r", encoding="utf-8"
    # ) as f:
    #     data_story = json.load(f)
    yield {
        "stage": "data_story",
        "dataset_name": dataset_name,
        "data_story": data_story,
        "message": "Data story generated successfully",
    }
    print(f"dataset_name:{dataset_name}, start to generate visualization template")
    html_template = InfographicTemplate(dataset_name, data_story).run()
    # with open(
    #     f"./results/{dataset_name}/infographic_template_1.html", "r", encoding="utf-8"
    # ) as f:
    #     html_template = f.read()
    yield {
        "stage": "html_template",
        "dataset_name": dataset_name,
        "html_template": html_template,
        "message": "HTML template generated successfully",
    }
    print(
        f"dataset_name:{dataset_name}, start to automatically generate data visualization and inv"
    )
    [inv, inv_no_data] = INV(
        dataset_name,
        json.loads(data_story),
        data_summary,
        html_template,
        data,
    ).run()
    # with open(f"./results/{dataset_name}/inv_1.html", "r", encoding="utf-8") as f:
    #     inv = f.read()
    yield {
        "stage": "inv",
        "dataset_name": dataset_name,
        "inv": inv,
        "message": "All processing completed",
    }
    print(f"dataset_name:{dataset_name}, start to evaluate inv")
    self_evaluation = Evaluate(dataset_name, inv_no_data).run()
    # with open(f"./results/{dataset_name}/evaluate_1.html", "r", encoding="utf-8") as f:
    #     self_evaluation = f.read()
    yield {
        "stage": "evaluation",
        "dataset_name": dataset_name,
        "self_evaluation": self_evaluation,
        "message": "All processing completed",
    }
    print(f"data_file:{data_file} end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, help="Dataset path")
    args = parser.parse_args()

    if args.datasets:
        dataset_path = args.datasets
        print(f"Dataset path: {dataset_path}")
        load_dotenv()
        d2inv(dataset_path)
    else:
        print("Dataset path parameter not provided")
