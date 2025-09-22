import os

import pandas as pd
from bs4 import BeautifulSoup

from tools.utils import filter_dataframe
from visualization import visualize_data_story


class INV:
    def __init__(
        self, dataset_name, data_story, data_summary, html_template, data: pd.DataFrame
    ):
        self.messages = []
        self.result = ""
        self.dataset_name = dataset_name
        self.data_story = data_story
        self.data_summary = data_summary
        self.html_template = html_template
        self.data = data

    def generate_interactive_narrative_visualization(self):
        soup = BeautifulSoup(self.html_template, "html.parser")
        script_tag = soup.new_tag(
            "script", src="https://unpkg.com/echarts@latest/dist/echarts.min.js"
        )
        soup.head.append(script_tag)
        script_tag = soup.new_tag(
            "script",
            src="https://unpkg.com/echarts@latest/dist/extension/dataTool.min.js",
        )
        soup.head.append(script_tag)
        # data_samples = (
        #     self.data.iloc[0:10].to_json(orient="records", force_ascii=False),
        # )
        visualization_codes = visualize_data_story(self.data_story, self.data_summary)
        for code in visualization_codes:
            code_soup = BeautifulSoup(code, "html.parser")
            code_style = code_soup.style.prettify()
            code_script = code_soup.script.prettify()

            soup.head.append(BeautifulSoup(code_style, "html.parser"))
            soup.body.append(BeautifulSoup(code_script, "html.parser"))

        inv_4_evaluate = soup.prettify()

        data = filter_dataframe(self.data)
        script_tag = soup.new_tag("script")
        script_tag.string = f"window.data = {data}"
        soup.head.append(script_tag)

        complete_inv = soup.prettify()

        return [
            complete_inv,
            inv_4_evaluate,
        ]

    def write(self, result):
        dist = f"./results/{self.dataset_name}"
        exist_count = len(
            list(filter(lambda x: x.startswith("inv_"), os.listdir(dist)))
        )
        with open(
            f"{dist}/inv_{exist_count+1}.html",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(result)

    def run(self):
        [inv, inv_no_data] = self.generate_interactive_narrative_visualization()
        self.write(inv)
        return inv_no_data
