import json

from tools.llm import get_llm
from tools.utils import preprocess_response


def evaluate_narration(data_summary, data_story):
    system_prompt = """
    You are a helpful assistant highly skilled in evaluating the quality of a given data story content by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER DATA STORYTELLING BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the data story across the following dimensions
    - Logic (logic): Do the insights flow logically from one section to the next? Are transitions between sections smooth and intuitive?
    - Structure (structure): Are the main insights divided into well-defined sections? Does the structure allow readers to easily follow the narrative?
    - Headings (headings): Are the headings engaging or intriguing enough to draw the reader's interest?Do the headings provide a clear overview of the story’s structure at a glance?
    - Subheadings (subheadings): Do subheadings provide clarity and guide readers through the story effectively? Do subheadings enhance readability and organization?
    - Narration in story pieces (narration): Does narration in each story piece contribute to the main narrative, or are any pieces unnecessary? Do the story pieces cover a diverse range of insights (e.g., trends, extremes, distributions, associations)?
    - Visualization in story pieces (visualization): CONSIDERING BEST PRACTICES, is the visualization type appropriate for the data and intent? Is there a visualization type that would be more effective in conveying insights? If a different visualization type is more appropriate, the score MUST be less than 5.

    You must provide a score for each of the above dimensions.

    Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format:
    [{ "dimension":  "logic",  "score": x , "rationale": " .."}, { "dimension":  "structure",  "score": x, "rationale": " .."}, { "dimension":  "headings",  "score": x, "rationale": " .."}, { "dimension":  "subheadings",  "score": x, "rationale": " .."}, { "dimension":  "narration",  "score": x, "rationale": " .."}, { "dimension":  "visualization",  "score": x, "rationale": " .."}
    ]
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "assistant",
            "content": f"Generate an evaluation given the data summary and data story. The data summary is \n\n {data_summary} \n\n and the data story is \n\n {data_story} \n\n . Now, evaluate the data story based on the 6 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS POOR AND A SCORE OF 10 IS VERY GOOD. The structured evaluation is below .",
        },
    ]
    [client, model] = get_llm()
    chat_completion = client.chat.completions.create(
        model=model, messages=messages, stream=False, temperature=1.3
    )
    result = chat_completion.choices[0].message.content
    result = preprocess_response(result)
    return json.loads(result)


def evaluate_data_visualization(goal, visualization_code):
    system_prompt = """
    You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER DATA VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions
    - bugs (bugs): Are there bugs, logic errors, syntax error or typos? Are there any reasons why the code may fail to compile? How should it be fixed? If ANY bug exists, the bug score MUST be less than 5.
    - Data transformation (transformation): Is the data transformed appropriately for the visualization type? E.g., is the dataset appropriated filtered, aggregated, or grouped  if needed?
    - Story piece compliance (compliance): How well the code meets the specified story pieces?
    - Data encoding (encoding): Is the data encoded appropriately for the visualization type?
    - aesthetics (aesthetics): Are the aesthetics of the visualization appropriate for the visualization type and the data?
    - interactions (interactions): Does the chart support interactive operations (e.g., hover, click, zoom, pan)?Does the chart provide visual feedback during interactions (e.g., highlighting selected data points)?
    
    You must provide a score for each of the above dimensions.  Assume that data in chart = plot(data) contains a valid dataframe for the dataset. The `plot` function returns a chart (e.g., echarts, d3 etc object).

    Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format:
    [{ "dimension":  "bugs",  "score": x , "rationale": " .."}, { "dimension":  "transformation",  "score": x, "rationale": " .."}, { "dimension":  "compliance",  "score": x, "rationale": " .."}, { "dimension":  "encoding",  "score": x, "rationale": " .."}, { "dimension":  "aesthetics",  "score": x, "rationale": " .."}, { "dimension":  "interactions",  "score": x, "rationale": " .."}]
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "assistant",
            "content": f"Generate an evaluation given the goal and code below in echarts. The specified goal is \n\n {goal.get('question')} \n\n and the code is \n\n {visualization_code} \n\n. Now, evaluate the code based on the 6 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS POOR AND A SCORE OF 10 IS VERY GOOD. The structured evaluation is below .",
        },
    ]
    [client, model] = get_llm()
    chat_completion = client.chat.completions.create(
        model=model, messages=messages, stream=False, temperature=1.3
    )
    result = chat_completion.choices[0].message.content
    result = preprocess_response(result)
    return json.loads(result)


def evaluate_inv(data_story, inv):
    system_prompt = """
    You are a helpful assistant highly skilled in evaluating the quality of a given interactive narrative visualization(INV) content by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER VISUAL DATA STORY BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the story across the following dimensions
    - Engagement (engagement): Does the visualization use interactive features (e.g., hover, click, zoom) to actively engage the user? Do click interactions (e.g., filtering, drilling down) allow users to dive deeper into the data story? Do interactive features (e.g., animations, transitions) evoke emotions (e.g., curiosity, surprise) that deepen user engagement? Do interactive elements align with the narrative flow, guiding users through the data story?
    - Usefulness (usefulness): Does INV effectively communicate the intended message or data story? Is the INV presented in a way that is easy to understand, even for non-expert audiences? Are the data sources clearly cited and reliable? Does the INV provide meaningful insights or teach the viewer something new?
    - Legibility (legibility): Is the text easy to read with appropriate font size, contrast, and spacing? Are the charts and visuals clear and easy to interpret? Is the hierarchy of information (e.g., headings, subheadings, story pieces) well-organized? Are the colors and typography consistent throughout the INV?
    - Design (design): Is the layout logical and intuitive, guiding the viewer through the information? Are the visual elements (e.g., icons, charts, images) aligned and balanced?Does the design use whitespace effectively to avoid clutter? Are the interactive elements (if any) well-integrated and functional?Does the design adapt well to different screen sizes and devices (responsive design)?
    - Aesthetics (aesthetics): Is the color scheme visually appealing and appropriate for the topic?Do the visuals (e.g., icons, illustrations, charts) enhance the overall aesthetic?Is the typography stylish yet readable? Are the animations or transitions (if any) smooth and visually pleasing? If there are no animations or transitions, the aesthetics score must be less than 5.
    You must provide a score for each of the above dimensions.

    Your final output format muse be:
    [{ "dimension":  "engagement",  "score": x , "rationale": " .."}, { "dimension":  "usefulness",  "score": x, "rationale": " .."}, { "dimension":  "legibility",  "score": x, "rationale": " .."}, { "dimension":  "design",  "score": x, "rationale": " .."}, { "dimension":  "aesthetics",  "score": x, "rationale": " .."}]
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "assistant",
            "content": f"Generate an evaluation given the data story and interactive narrative visualization(INV) in html format. The data story is \n\n {data_story} \n\n and the INV in html format is \n\n {inv} \n\n. Now, evaluate the INV based on the 5 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS POOR AND A SCORE OF 10 IS VERY GOOD. The structured evaluation is below .",
        },
    ]
    [client, model] = get_llm()
    chat_completion = client.chat.completions.create(
        model=model, messages=messages, stream=False, temperature=1.3
    )
    result = chat_completion.choices[0].message.content
    result = preprocess_response(result)
    print(result)
    return json.loads(result)


def evaluation_visualize(evaluate_result):
    system_prompt = """Generate an HTML visualization using ECharts that displays the given JSON data in a radar chart. I also want interactive elements on the right side, with each dimension showing a star rating and progress bar in one row. The height of radar chart and  interactive elements should be equal. Additionally, when a dimension is selected, its details should appear while others disappear. """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "assistant",
            "content": f"The JSON data is: \n {evaluate_result} \n The generated HTML is: \n",
        },
    ]
    [client, model] = get_llm()
    chat_completion = client.chat.completions.create(
        model=model, messages=messages, stream=False, temperature=1.3
    )
    result = chat_completion.choices[0].message.content
    result = preprocess_response(result)
    return result
