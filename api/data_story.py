import json
from tools.llm import get_llm
from tools.utils import preprocess_response
from pathlib import Path


def generate_data_story(
    file_content, vendor="openai", model_type="general", data_fact_max=5
):
    system_prompt = f"""
        ## Profile:
        - Author: D2INV
        - Version: 1.0
        - Language: English
        - Specialization: An experienced data analyst skilled in generating coherent and engaging data stories.
        
        ## Goals:
        Generate a data story from the provided dataset content.
        
        ## Constraints:
        1. The content includes a story_title, a story_subtitle, and a story_pieces list. 
        2. Each element in the story_pieces list contains a question (what question this story piece answers), a narration (the data facts corresponding to this story fragment), and a visualization (what type of visualization is suitable for presenting this question and data fact). 
        3. The story_pieces should have valid logic and structure. Logic refers to the relationships between data facts, while structure refers to the good organization of the entire narrative content. 
        4. The generated data story content should be returned in JSON format.
        5. The number of story pieces should not exceed {data_fact_max}
        ## Skills:
        - Expertise in data analysis and storytelling.
        - Proficiency in identifying key insights and crafting coherent narratives.
        
        ## Workflow:
        1. **Understand the Data**:
           - Analyze the dataset to identify its core components (e.g., entities, metrics, timeframes).
           - Example: "The dataset contains [Data Type] records with fields like [Field 1], [Field 2], and [Field 3]."
        2. **Identify Key Insights**:
           - Group data into logical categories (e.g., regions, categories, time periods).
           - Example: "The data spans [Number] [Time Unit] and includes [Number] unique [Entities]."
        3. **Formulate Questions**:
           - Develop specific, answerable questions about the data.
           - Example:
                a) "Which [Entity] had the highest [Metric]?"
                b) "How did [Metric] change over [Timeframe]?"
                c) "What patterns exist between [Metric 1] and [Metric 2]?"
        4. **Narrate Data Facts**:
           - For each question, summarize the relevant data insights.
           - Example: "Entity X showed a [Trend] in [Metric], with a peak of [Value] on [Date/Condition]."
        5. **Select Visualizations**:
           - Match each question to an appropriate chart type (e.g., bar charts for comparisons, line charts for trends)..
           - Example: "A heatmap would highlight correlations between [Metric 1] and [Metric 2]."
        6. **Structure the Story**:
           - Organize story pieces from broad overviews to specific insights.
           - Example Flow: Overview of dataset → Top performers → Key trends → Comparative analysis → Outliers
        7. **Output the Data Story**:
           - Final Output Format:
           {{  
              "story_title": "[Descriptive Title]",  
              "story_subtitle": "[Brief Summary]",  
              "story_pieces": [  
                {{  
                  "question": "[Specific Question]",  
                  "narration": "[Data-Driven Insight]",  
                  "visualization": "[Chart Type and Rationale]"  
                }},  
                ...  
              ]  
            }}
        """
    [client, model] = get_llm(vendor=vendor, model_type=model_type)
    # file_object = client.files.create(file=Path(file_path), purpose="file-extract")
    # file_content = client.files.content(file_id=file_object.id).text
    messages = (
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": file_content,
            },
            {
                "role": "user",
                "content": f"The generated data story is: \n ",
            },
        ]
        if model_type == "general"
        else [
            {
                "role": "system",
                "content": "Generate a data story based on the provided JSON or CSV file. The content includes a story_title, a story_subtitle, and a story_pieces list. Each element in the story_pieces list contains a question (what question this story piece answers), a narration (the data facts corresponding to this story fragment), and a visualization (what type of visualization is suitable for presenting this question and data fact). The story_pieces should have valid logic and structure. Logic refers to the relationships between data facts, while structure refers to the good organization of the entire narrative content. The generated data story content should be returned in JSON format.",
            },
            {
                "role": "system",
                "content": file_content,
            },
            {
                "role": "user",
                "content": f"The generated data story is: \n",
            },
        ]
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=1.0,
    )
    result = chat_completion.choices[0].message.content
    result = preprocess_response(result)
    return result


def write_data_story(dataset_name, index, result):
    with open(
        f"../evaluate_results/{dataset_name}/data_story_{index}.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(result)
