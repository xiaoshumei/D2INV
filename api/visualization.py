import os

from tools.llm import LLM, RequestsLLM
from tools.utils import postprocess_response


class Visualization:

    def __init__(self, index, goal, library="echarts"):
        self.messages = []
        self.result = ""
        self.index = index
        self.goal = goal
        self.llm = LLM()
        self.library = library
        self.requests_llm = RequestsLLM()

    def reason(
        self,
        data_summary,
    ):
        """
        Generate a data story from the provided dataset using reasoning llm.

        Returns:
            str: The generated data story in JSON format.
        """
        index = self.index
        print("story piece: ", index, "start to visualize")
        visualization = self.goal.get("visualization")
        question = self.goal.get("question")
        template = f"""
        <style>
            #chart_{index + 1} {{
                <stub> // The first `<stub>` section
            }}
            <stub> // The second `<stub>` section
        </style>
        <script>
            function plot_{index + 1}(data){{
                document.getElementById('chart_{index + 1}').innerHTML = '';
                const myChart = {self.library}.init(document.getElementById('chart_{index + 1}'), null, {{ renderer: 'svg' }});

                <stub> // The third `<stub>` section
            }}
            plot_{index + 1}(data) // data already contains the data to be plotted. Always include this line. No additional code beyond this line.
        </script>
        """
        self.messages += [
            {
                "role": "system",
                "content": f"""Please generate a piece of HTML code that only contains a script tag and a style tag. Based on the given code template and data summary, use {self.library} to generate visualization code. The code template has three <stub> placeholders. Only add new code where the placeholders are and remove them. Don't modify any other part of the code. The data required for the chart has been extracted and injected into the `window.data` variable. Please carefully read the provided data summary, which includes information such as the data's columns, data type, theme, and so on.""",
            },
            {
                "role": "system",
                "content": f"The data summary is\n:{data_summary}\n The code template is: {template} \n All mathematical calculation results must retain only two decimal places. When generating code that requires any map, you must load the corresponding GeoJSON file only from:\n ```js\nfetch('/tools/maps/<filename>')\nwhere <filename> is one of the exact names listed below:\n{os.listdir('./tools/maps')}\nDo not use any other path, CDN, or additional checks/explanations.\n",
            },
            {
                "role": "user",
                "content": f"Generate a {self.library} chart ({visualization}) that addresses this goal: {question}.",
            },
        ]

        if os.getenv("USE_OPENAI") == "False":
            content = self.requests_llm.chat(self.messages)
        else:
            completion = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=self.messages,
                stream=False,
                temperature=1.0,
            )
            content = completion.choices[0].message.content
        print("reason\n", content)
        # add response into chat history
        self.messages.append({"role": "assistant", "content": content})

    def reflection(self):
        self.messages += [
            {
                "role": "user",
                "content": f"Reflect on whether there are any issues with the above visualization code—for example, syntax errors, references to columns not mentioned in the data summary, incorrect usage of column value types, reliance on deprecated {self.library} APIs, CSS in the style tag that alters ancestor elements’ styles, or failure to round mathematical calculation results to two decimal places.",
            },
        ]
        if os.getenv("USE_OPENAI") == "False":
            content = self.requests_llm.chat(self.messages)
        else:
            completion = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=self.messages,
                stream=False,
                temperature=1.0,
            )
            content = completion.choices[0].message.content
        print("reflection\n", content)

        # add response into chat history
        self.messages.append({"role": "assistant", "content": content})

    def refine(self):
        self.messages += [
            {
                "role": "user",
                "content": "Fix the above issues and provide a new visualization code. Don't do any explanation.",
            },
        ]
        if os.getenv("USE_OPENAI") == "False":
            content = self.requests_llm.chat(self.messages)
        else:
            completion = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=self.messages,
                stream=False,
                temperature=1.0,
            )
            content = completion.choices[0].message.content
        print("refine\n", content)
        # TODO: Detect whether the chart height is invalid, such as unset height, height of 0, or height of 100%
        self.result = postprocess_response(content)

    def run(self, data_summary):
        self.reason(data_summary)
        self.reflection()
        self.refine()
        return self.result


def visualize_data_story(data_story, data_summary):
    visualization_codes = []
    for index, goal in enumerate(data_story.get("story_pieces")):
        code = Visualization(index, goal).run(data_summary)
        visualization_codes.append(code)
    return visualization_codes
