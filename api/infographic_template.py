from tools.llm import LLM
from tools.utils import postprocess_response
import os
from bs4 import BeautifulSoup


class InfographicTemplate:
    def __init__(self, dataset_name, data_story):
        self.messages = [
            {
                "role": "system",
                "content": """Please design an infographic template in HTML format. First, carefully read the given data story. Then, choose an infographic layout and structure that match the story. Finally, create an attractive infographic template for narrative visualization. 
                The given data story will includes story_title, story_subtitle, and multiple story_pieces. Each story_piece contains narration (the discovered data facts), question (the question posed about the data fact), and visualization (the chart type that best visualizes the data fact for the question). 
                The template must include a title (corresponding to the data story's story_title), a subtitle (corresponding to the data story's story_subtitle), and multiple sections to display all story_pieces.  
                For each section, the template must display its title (corresponding to the story_piece's question), description (corresponding to the story_piece's narration), and an empty div element with a unique id.
                The unique id must with the value chart_index, where index is replaced by the sequence number of the story_piece, starting from 1(chart_1 for the first, chart_2 for the second, and so on). 
                Do not assign any height to the empty div element, it will serve as the placeholder where the charts will be rendered in the next step, and its height will be determined by its children.
                The template must adhere to infographic best practices, such as: 
                (1) The infographic must include:
                    a. Explainer: Brief, non-sentence introductory text. 
                    b. Sections: Clearly separated using lines, icons, or alternating colors. 
                    c. Explanatory Text: Support graphics with concise descriptions. 
                    d. Call to Action (CTA): Encourage readers to take the next step. 
                    e. Headline: Create a compelling and concise title. 
                    f. Subheads/Labels: Clarify sections, icons, and charts. 
                    g. Backbone: Unifying design element (line, icon, visual) to guide readers. 
                    h. Sources: Include brief citations at the end.  
                (2) The style of the infographic should match the content of the data story. 
                (3) The infographic should use rich colors, icons (e.g., from Icons8; ensure the icon links are legal and accessible), and animation effects (e.g., the AOS library). 
                (4) Add Calls to Action between different sections to guide users to continue browsing.""",
            }
        ]
        self.result = ""
        self.dataset_name = dataset_name
        self.data_story = data_story
        self.llm = LLM()

    def reason(self):
        self.messages.append(
            {
                "role": "user",
                "content": f"The content of the data story is: \n {self.data_story} \n Only return the HTML content and Don't explain anything. The generated html infographic is:  \n ",
            }
        )

        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=self.messages,
            stream=False,
            temperature=1.0,
        )
        content = completion.choices[0].message.content
        # add response into chat history
        self.messages.append({"role": "assistant", "content": content})

    def reflection(self):
        self.messages += [
            {
                "role": "user",
                "content": "Analyze the issues in the above html infographic. Check if each story_piece has an empty div element with a unique id but without height set through external CSS or inline CSS. Besides, reflect on whether the aesthetic indicators—color, font, icons, animations, layout, and structure—meet the required standards.",
            },
        ]
        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=self.messages,
            stream=False,
            temperature=1.0,
        )
        content = completion.choices[0].message.content
        # add response into chat history
        self.messages.append({"role": "assistant", "content": content})

    def refine(self):
        self.messages += [
            {
                "role": "user",
                "content": "Fix the above issues and provide a new infographic-style template in HTML format. Don't do any explanation.",
            },
        ]
        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=self.messages,
            stream=False,
            temperature=1.0,
        )
        content = completion.choices[0].message.content
        self.result = postprocess_response(content)
        # Parse the HTML template string into a BeautifulSoup object for manipulation
        soup = BeautifulSoup(self.result, "html.parser")

        self.result = soup.prettify()

    def write(self):
        dist = f"./results/{self.dataset_name}"
        exist_count = len(
            list(
                filter(
                    lambda x: x.startswith("infographic_template_"), os.listdir(dist)
                )
            )
        )
        with open(
            f"{dist}/infographic_template_{exist_count+1}.html",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(self.result)

    def run(self):
        self.reason()
        self.reflection()
        self.refine()
        self.write()
        return self.result

    def edit(self, prompts):
        """
        Modify the existing infographic template based on user input prompts
        """
        messages = [
            {
                "role": "system",
                "content": """You are an infographic template editor. Your task is to modify the existing HTML infographic template based on user prompts while maintaining proper HTML structure and functionality. The template includes a title, subtitle, multiple sections with charts, and follows infographic best practices. Make sure to preserve the overall structure and return valid HTML.""",
            },
            {
                "role": "user",
                "content": f"The current infographic template is: {self.result}. The user requests the following modifications: {prompts}. Please update the infographic template accordingly and return the modified version in valid HTML format. Do not include any explanations or comments in the HTML. Ensure that all existing chart div elements with ids like chart_1, chart_2, etc. are preserved unless specifically requested to modify them.",
            },
        ]

        completion = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            stream=False,
            temperature=0.7,
        )
        content = completion.choices[0].message.content

        content = postprocess_response(content)

        # remove content after </html> tag
        content = content[: content.find("</html>")]
        print("edit:\n", content)

        # Parse the HTML template string into a BeautifulSoup object for manipulation
        soup = BeautifulSoup(content, "html.parser")
        self.result = soup.prettify()

        return self.result
