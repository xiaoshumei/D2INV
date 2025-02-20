import os
from summarize import simple_summary, write_summary
from data_story import generate_data_story, write_data_story
from html_template_generate import generate_html_template, write_html_template
from generate_inv import (
    generate_interactive_narrative_visualization,
    write_interactive_narrative_visualization,
)
import json
from tools.utils import data_dir
from evaluate import evaluate_narration, evaluate_data_visualization, evaluate_inv


def get_date_files():
    data_list = os.listdir(data_dir)
    return data_list


def experiments():
    repeat_time = 1
    data_list = get_date_files()
    data_list = ["disasters.csv"]
    for data_file in data_list:
        print(f"data_file:{data_file} start")
        dataset_name = data_file.split(".")[0]
        os.makedirs(f"../evaluate_results/{dataset_name}", exist_ok=True)
        current = 1
        evaluate_results = {
            "data_story": [],
            "data_visualization": [],
            "inv": [],
        }
        while current <= repeat_time:
            print(f"dataset_name:{dataset_name} sequence:{current} start to summarize")
            [data_summary, data] = simple_summary(data_file)
            [cleaned_df, split_df] = data
            write_summary(dataset_name, current, json.dumps(data_summary))
            file_content = split_df.to_json(
                orient="records", lines=False, force_ascii=False
            )
            print(
                f"dataset_name:{dataset_name} sequence:{current} start to generate data story"
            )
            data_story = generate_data_story(
                file_content, vendor="openai", model_type="general"
            )
            write_data_story(dataset_name, current, data_story)
            evaluation_data_story = evaluate_narration(data_summary, data_story)
            evaluate_results["data_story"].append(evaluation_data_story)
            print(
                f"dataset_name:{dataset_name} sequence:{current} start to generate visualization template"
            )
            html_template = generate_html_template(
                data_story, vendor="openai", model_type="general"
            )
            write_html_template(dataset_name, current, html_template)
            print(
                f"dataset_name:{dataset_name} sequence:{current} start to automatically data visualization"
            )
            [inv, inv_no_data, evaluation_visualization] = (
                generate_interactive_narrative_visualization(
                    json.loads(data_story),
                    data_summary,
                    html_template,
                    cleaned_df,
                    vendor="openai",
                    model_type="general",
                )
            )
            evaluate_results["data_visualization"].append(evaluation_visualization)
            write_interactive_narrative_visualization(dataset_name, current, inv)
            evaluation_inv = evaluate_inv(data_story, inv_no_data)
            evaluate_results["inv"].append(evaluation_inv)
            print(
                f"dataset_name:{dataset_name} sequence:{current} start to generate inv"
            )
            with open(
                f"../evaluate_results/{dataset_name}/evaluation_result_{current}.json",
                "w",
            ) as f:
                json.dump(evaluate_results, f)
            current += 1
        print(f"data_file:{data_file} end")


if __name__ == "__main__":
    experiments()
