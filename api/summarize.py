import json
import warnings
from json import JSONEncoder

import pandas as pd

from tools.utils import read_dataframe


def check_type(dtype, value):
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
            properties["mean"] = check_type(dtype, df[column].mean())
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
        num_unique = df[column].nunique()
        if "samples" not in properties:
            non_null_values = df[column][df[column].notnull()].unique()
            n_samples = min(n_samples, len(non_null_values))
            samples = (
                pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
            )
            properties["samples"] = samples
        properties["num_unique_values"] = num_unique
        if properties["dtype"] == "category":
            # Add distribution information of each unique value when datatype is category
            properties["distribution_unique_values"] = (
                df[column].value_counts().to_dict()
            )
        properties_list.append({"column": column, "properties": properties})

    return properties_list


def file_summary(file_name):
    data = read_dataframe(file_name)
    n_samples = 3
    data_properties = get_column_properties(data, n_samples)

    # construct the base data summary
    base_summary = {
        "name": file_name,
        "fields": data_properties,
    }
    return [base_summary, data]


def batch_summary(batch_data):
    n_samples = 3
    batch_data = pd.DataFrame(batch_data)
    data_properties = get_column_properties(batch_data, n_samples)

    # construct the base data summary
    summary = {
        "fields": data_properties,
    }
    return summary


class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def write_summary(dataset_name, result):
    data_summary = json.dumps(result, cls=CustomEncoder)
    dist = f"./results/{dataset_name}"
    with open(
        f"{dist}/data_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(data_summary)
