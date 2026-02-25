import os
from tools.llm import LLM
from tools.utils import postprocess_response


class Evaluate:
    def __init__(self, dataset_name, inv):
        self.dataset_name = dataset_name
        self.inv = inv
        self.llm = LLM()

    def evaluate_inv(self):
        system_prompt = """
        You are a helpful assistant highly skilled in evaluating the quality of a given interactive narrative visualization(INV) content by providing a score from 1 (bad) - 10 (good) while providing clear rationale. 
        YOU MUST CONSIDER VISUAL DATA STORY BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the story across the following dimensions
        - Engagement (engagement): Does the visualization use interactive features (e.g., hover, click, zoom) to actively engage the user? Do click interactions (e.g., filtering, drilling down) allow users to dive deeper into the data story? Do interactive features (e.g., animations, transitions) evoke emotions (e.g., curiosity, surprise) that deepen user engagement? Do interactive elements align with the narrative flow, guiding users through the data story?
        - Usefulness (usefulness): Does INV effectively communicate the intended message or data story? Is the INV presented in a way that is easy to understand, even for non-expert audiences? Are the data sources clearly cited and reliable? Does the INV provide meaningful insights or teach the viewer something new?
        - Legibility (legibility): Is the text easy to read with appropriate font size, contrast, and spacing? Are the charts and visuals clear and easy to interpret? Is the hierarchy of information (e.g., headings, subheadings, story pieces) well-organized? Are the colors and typography consistent throughout the INV?
        - Design (design): Is the layout logical and intuitive, guiding the viewer through the information? Are the visual elements (e.g., icons, charts, images) aligned and balanced?Does the design use whitespace effectively to avoid clutter? Are the interactive elements (if any) well-integrated and functional?Does the design adapt well to different screen sizes and devices (responsive design)?
        - Aesthetics (aesthetics): Is the color scheme visually appealing and appropriate for the topic?Do the visuals (e.g., icons, illustrations, charts) enhance the overall aesthetic?Is the typography stylish yet readable? Are the animations or transitions (if any) smooth and visually pleasing? If there are no animations or transitions, the aesthetics score must be less than 5.
        You must provide a score for each of the above dimensions.

        The JSON you ultimately output must conform to the following format:
        [
            { "dimension":  "engagement",  "score": x , "rationale": " .."}, 
            { "dimension":  "usefulness",  "score": x, "rationale": " .."}, 
            { "dimension":  "legibility",  "score": x, "rationale": " .."}, 
            { "dimension":  "design",  "score": x, "rationale": " .."}, 
            { "dimension":  "aesthetics",  "score": x, "rationale": " .."}
        ]
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": f"Generate an evaluation given the data story and interactive narrative visualization(INV) in html format. The INV in html format is \n\n {self.inv} \n\n. Now, evaluate the INV based on the 5 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS POOR AND A SCORE OF 10 IS VERY GOOD. The structured evaluation is below .",
            },
        ]

        chat_completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            stream=False,
            temperature=1.0,
            response_format={"type": "json_object"},
        )
        content = chat_completion.choices[0].message.content
        return postprocess_response(content)

    def evaluation_visualize(self, evaluate_result):
        system_prompt = """Generate an HTML visualization using ECharts that displays the given JSON data in a radar chart. I also want interactive elements on the right side, with each dimension showing a star rating and progress bar in one row. The height of radar chart and  interactive elements should be equal. Additionally, when a dimension is selected, its details should appear while others disappear. Only return the HTML content and don't explain anything"""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"The JSON data is: \n {evaluate_result} \n The generated HTML is: \n",
            },
        ]

        chat_completion = self.llm.client.chat.completions.create(
            model=self.llm.model, messages=messages, stream=False, temperature=1.0
        )
        evaluate_visualize_result = chat_completion.choices[0].message.content
        result = postprocess_response(evaluate_visualize_result)
        # remove content after </html> tag
        result = result[: result.find("</html>")]
        self.write(result)
        return result

    def write(self, result):
        dist = f"./results/{self.dataset_name}"
        exist_count = len(
            list(filter(lambda x: x.startswith("evaluate_"), os.listdir(dist)))
        )
        with open(
            f"{dist}/evaluate_{exist_count + 1}.html",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(result)

    def run(self):
        evaluate_result = self.evaluate_inv()
        return self.evaluation_visualize(evaluate_result)
