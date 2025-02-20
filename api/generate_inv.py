import pandas as pd
from tools.llm import get_llm
from bs4 import BeautifulSoup
from tools.utils import preprocess_response, filter_dataframe

# from evaluate import evaluate_data_visualization


def generate_interactive_narrative_visualization(
    data_story,
    data_summary,
    html_template,
    data_frame: pd.DataFrame,
    library="echarts",
    vendor="openai",
    model_type="general",
):
    system_prompt = f"""
        ## Profile:
        - Author: D2INV
        - Version: 1.0
        - Language: English
        - Specialization: A highly skilled assistant in writing PERFECT code for data visualizations, ensuring best practices and accuracy.
        
        ## Goals:
        1. Complete provided code templates to generate visualization codes based on a given goal.
        2. Generate the visualization by using {library}.
        3. Carefully solve the task by completing three `<stub>` sections. Think step-by-step 
        
        ## Constraints:
        1. **Best Practices**: 
           - The code MUST follow visualization best practices (e.g., correct visualization type, proper data encoding, legible aesthetics, and clear axis labels).
           - Ensure the visualization meets the specified goal and uses the right transformations.
        2. **Correctness**:
           - The code MUST be free of syntax or logic errors.
           - Use fields and transformations correctly, considering field types (e.g., numeric, date, categorical).
        4. **Specific Requirements**:
           - If the solution requires a single value (e.g., max, min, median), ALWAYS add a line (`axvline` or `axhline`) to the chart with a legend displaying the value (formatted to `0.2f`).
           - If the solution requires a geo map, make sure the map data is correctly registered before setting the option.
        5. **Legibility**:
           - Ensure that the legend and title do not overlap with the chart to avoid obscuring it.
           - Ensure x-axis labels are legible (e.g., rotate if necessary).
           - When there are many categories on the X-axis, hide the labels of the chart.
        6. **Data Usage**:
           - ONLY use fields that exist in the dataset or are derived from existing fields.
           - DO NOT write code to load data; assume the data is already loaded in the variable `data`.
        7. **ECharts**:
           - ALWAYS use the latest version of ECharts API.
           - Add legends with appropriate colors where necessary.
        8. **CSS**:
           - Use ID selectors (e.g., `#chart_{{index+1}}`) for CSS classes to prevent naming conflicts.
        9. **Precision**:
           - Round all mathematical operations to two decimal places.
        10. **Output**:
            - ONLY return the modified `<style>` and `<script>` tags.
            - DO NOT include any explanations or additional text.
        
        ## Skills:
        - Expertise in writing clean, efficient, and accurate code for data visualizations.
        - Deep understanding of visualization best practices and ECharts API.
        
        ## Workflow:
        1. **Understand the Goal**:
           - Read and fully comprehend the visualization goal.
        2. **Plan the Solution**:
           - Generate a brief plan for the solution, including:
             - Required transformations (e.g., new columns, aggregations).
             - Visualization type (e.g., bar chart, line chart).
             - Aesthetics (e.g., colors, axis formatting).
             - Constraints (Identify and implement all of the user's constraints one by one).
        3. **Complete the Code**:
           - Fill in the third `<stub>` section in the provided template.
           - Ensure the code is functional and adheres to best practices.
           - Round all mathematical results to two decimal places.
           - Do not delete any existing code  in the template except for the <stub> itself.
        4. **Add Styles**:
           - Set specific and accurate width and height for the chart in the first `<stub>` section.
           - Generate necessary CSS styles in the second `<stub>` section.
           - Use ID selectors (e.g., `#chart_{{index+1}}`) to avoid conflicts.
        5. **Finalize**:
           - Remove all three `<stub>` sections from the template.
        6. **Return Output**:
           - ONLY return the modified `<style>` and `<script>` tags.
    """

    soup = BeautifulSoup(html_template, "html.parser")
    script_tag = soup.new_tag(
        "script", src="https://unpkg.com/echarts@latest/dist/echarts.min.js"
    )
    soup.head.append(script_tag)
    script_tag = soup.new_tag(
        "script", src="https://unpkg.com/echarts@latest/dist/extension/dataTool.min.js"
    )
    soup.head.append(script_tag)
    evaluate_results = []
    for index, goal in enumerate(data_story.get("story_pieces")):
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
                const myChart = {library}.init(document.getElementById('chart_{index + 1}'), null, {{ renderer: 'svg' }});
                
                <stub> // The third `<stub>` section
            }}
            plot_{index + 1}(data) // data already contains the data to be plotted. Always include this line. No additional code beyond this line.
        </script>
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": f"The dataset summary is : {data_summary} \n\n The code template is: {template} \n\n",
            },
            {
                "role": "user",
                "content": f"Generate a {library} chart ({goal.get('visualization')}) that addresses this goal: {goal.get('question')}. The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ... \n\n",
            },
        ]
        [client, model] = get_llm(vendor="openai", model_type="general")
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
        )

        code = chat_completion.choices[0].message.content
        code = preprocess_response(code)

        evaluate_results.append(evaluate_data_visualization(goal, code))
        code_soup = BeautifulSoup(code, "html.parser")
        code_style = code_soup.style.prettify()
        code_script = code_soup.script.prettify()

        soup.head.append(BeautifulSoup(code_style, "html.parser"))
        soup.body.append(BeautifulSoup(code_script, "html.parser"))

    inv_no_data = soup.prettify()

    data = filter_dataframe(data_summary, data_frame)
    # data = str(data).replace("Timestamp", "new Date")
    script_tag = soup.new_tag("script")
    script_tag.string = f"window.data = {data}"
    soup.head.append(script_tag)  # 插入到 <head> 的第一个位置

    complete_inv = soup.prettify()

    return [complete_inv, inv_no_data, evaluate_results]


def write_interactive_narrative_visualization(dataset_name, index, result):
    with open(
        f"../evaluate_results/{dataset_name}/inv_{index}.html",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(result)
