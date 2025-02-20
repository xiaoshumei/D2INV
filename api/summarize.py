import json
from tools.llm import get_llm
import pandas as pd
import warnings
from tools.utils import read_dataframe, preprocess_response


def check_type(dtype: str, value):
    """Cast value to right type to ensure it is JSON serializable"""
    if "float" in str(dtype):
        return float(value)
    elif "int" in str(dtype):
        return int(value)
    else:
        return value


def get_column_properties(df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
    """Get properties of each column in a pandas DataFrame"""
    properties_list = []
    for column in df.columns:
        dtype = df[column].dtype
        properties = {}
        if dtype in [int, float, complex]:
            properties["dtype"] = "number"
            properties["std"] = check_type(dtype, df[column].std())
            properties["min"] = check_type(dtype, df[column].min())
            properties["max"] = check_type(dtype, df[column].max())

        elif dtype == bool:
            properties["dtype"] = "boolean"
        elif dtype == object:
            # Check if the string column can be cast to a valid datetime
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(df[column], errors="raise")
                    properties["dtype"] = "date"
            except ValueError:
                # Check if the string column has a limited number of values
                if df[column].nunique() / len(df[column]) < 0.5:
                    properties["dtype"] = "category"
                else:
                    properties["dtype"] = "string"
        elif isinstance(df[column], pd.CategoricalDtype):
            properties["dtype"] = "category"
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            properties["dtype"] = "date"
        else:
            properties["dtype"] = str(dtype)

        # add min max if dtype is date
        if properties["dtype"] == "date":
            try:
                properties["min"] = df[column].min()
                properties["max"] = df[column].max()
            except TypeError:
                cast_date_col = pd.to_datetime(df[column], errors="coerce")
                properties["min"] = cast_date_col.min()
                properties["max"] = cast_date_col.max()
        # Add additional properties to the output dictionary
        nunique = df[column].nunique()
        if "samples" not in properties:
            non_null_values = df[column][df[column].notnull()].unique()
            n_samples = min(n_samples, len(non_null_values))
            samples = (
                pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
            )
            properties["samples"] = samples
        properties["num_unique_values"] = nunique
        properties["semantic_type"] = ""
        properties["description"] = ""
        properties_list.append({"column": column, "properties": properties})

    return properties_list


def cot_summary(file_name):
    data = read_dataframe(file_name)
    n_samples = 3
    data_properties = get_column_properties(data, n_samples)

    # construct the base data summary
    base_summary = {
        "name": file_name,
        "file_name": file_name,
        "description": "",
        "fields": data_properties,
    }
    system_prompt = """
        ## Profile:
        - Author: D2INV
        - Version: 1.0
        - Language: English
        - Description: An experienced data analyst skilled in annotating and summarizing datasets.
        
        ## Goals:
        Enrich the base summary by providing detailed descriptions and semantic types for each field without altering the existing structure.
           
        ## Skills:
        - Expertise in data analytics and dataset annotation.
        - Familiarity with a wide range of semantic types and their applications.
        
        ## Constraints
        - JSON requires double quotes for keys and string values.
        
        ## Workflow:
        1. First, You need to understand the dataset.
        2. Starting with the dataset's name and description. You should come up with a concise name that reflects the content. The description should elaborate on what the dataset contains,
        3. Next, each field in the 'fields' array needs a description and semantic type(e.g., company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc.). You must go through them one by one.
        4. You need to ensure that all fields are covered, descriptions are clear, and semantic types are appropriate. Also, the dataset's overall name and description should be accurate and informative. 
        5. Finally, you put this all together in the JSON format, making sure not to include any extra text or omit any fields.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Annotate the base data summary below.\n{base_summary}""",
        },
    ]
    [client, model] = get_llm()
    chat_completion = client.chat.completions.create(
        model=model, messages=messages, stream=False
    )
    result = chat_completion.choices[0].message.content
    result = preprocess_response(result)
    result = json.loads(result)
    return [result, data]


def simple_summary(file_name):
    data = read_dataframe(file_name)
    [cleaned_df, _] = data
    n_samples = 3
    data_properties = get_column_properties(cleaned_df, n_samples)

    # construct the base data summary
    base_summary = {
        "name": file_name,
        "file_name": file_name,
        "description": "",
        "fields": data_properties,
    }
    return [base_summary, data]


def write_summary(dataset_name, index, result):
    with open(
        f"../evaluate_results/{dataset_name}/data_summary_{index}.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(result)
