import os
import re
import pandas as pd
import logging
from tools.config import data_dir

logger = logging.getLogger("D2INV")


def clean_column_name(col_name: str) -> str:
    """
    Clean a single column name by replacing special characters and spaces with underscores.

    :param col_name: The name of the column to be cleaned.
    :return: A sanitized string valid as a column name.
    """
    return re.sub(r"[^0-9a-zA-Z_]", "_", col_name)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all column names in the given DataFrame.

    :param df: The DataFrame with possibly dirty column names.
    :return: A copy of the DataFrame with clean column names.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = [clean_column_name(col) for col in cleaned_df.columns]
    return cleaned_df


def clean_nan_rows(df: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
    """
    Clean rows with NaN values in the given DataFrame.

    :param df: The DataFrame with possibly NaN values.
    :param numerical_columns: The columns with number type
    :return: A copy of the DataFrame with NaN values removed.
    """
    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna(subset=numerical_columns)
    return cleaned_df


def postprocess_response(response):
    """
    Remove code block markers (```xxx) from LLM outputs
    Keep only the inner content
    Works for both complete (with closing ```)
    and incomplete blocks (without closing ```)
    """
    # Remove starting ``` with optional language tag (e.g., ```html, ```json, ```python)
    text = re.sub(r"^```\w*\n?", "", response, flags=re.MULTILINE)
    # Remove ending ```
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    return text.strip()


def filter_dataframe(df: pd.DataFrame) -> str:
    """
    Filter the dataframe by dropping the nan values.
    :param df: data frame to be filtered
    :return: data frame after filter
    """
    numerical_columns = [
        column for column in df.columns if df[column].dtype in [int, float, complex]
    ]
    df = clean_nan_rows(df, numerical_columns)
    data = df.to_json(orient="records", force_ascii=False)
    return data


def read_dataframe(file_name, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Read a dataframe from a given file location and clean its column names.
    It also samples down to 4500 rows if the data exceeds that limit.

    :param file_name: The name of the data file.
    :param encoding: Encoding to use for the file reading.
    :return: A cleaned DataFrame.
    """
    file_location = str(os.path.join(data_dir, file_name))
    file_extension = file_location.split(".")[-1]
    print(file_location)
    read_funcs = {
        "json": lambda: pd.read_json(
            file_location, orient="records", encoding=encoding, convert_dates=False
        ),
        "jsonl": lambda: pd.read_json(
            file_location,
            orient="records",
            encoding=encoding,
            convert_dates=False,
            lines=True,
        ),
        "csv": lambda: pd.read_csv(
            file_location,
            encoding=encoding,
            keep_default_na=False,
        ),
        "xls": lambda: pd.read_excel(file_location),
        "xlsx": lambda: pd.read_excel(file_location),
        "parquet": lambda: pd.read_parquet(file_location, engine="pyarrow"),
        "feather": lambda: pd.read_feather(file_location),
        "tsv": lambda: pd.read_csv(file_location, sep="\t", encoding=encoding),
    }
    if file_extension not in read_funcs:
        raise ValueError("Unsupported file type")

    try:
        df = read_funcs[file_extension]()

    except Exception as e:
        logger.error(f"Failed to read file: {file_location}. Error: {e}")
        raise

    # Clean column names
    cleaned_df = clean_column_names(df)

    if cleaned_df.columns.tolist() != df.columns.tolist():
        write_funcs = {
            "csv": lambda: cleaned_df.to_csv(
                file_location, index=False, encoding=encoding
            ),
            "xls": lambda: cleaned_df.to_excel(file_location, index=False),
            "xlsx": lambda: cleaned_df.to_excel(file_location, index=False),
            "parquet": lambda: cleaned_df.to_parquet(file_location, index=False),
            "feather": lambda: cleaned_df.to_feather(file_location, index=False),
            "json": lambda: cleaned_df.to_json(
                file_location, orient="records", index=False, default_handler=str
            ),
            "tsv": lambda: cleaned_df.to_csv(
                file_location, index=False, sep="\t", encoding=encoding
            ),
        }

        if file_extension not in write_funcs:
            raise ValueError("Unsupported file type")

        try:
            write_funcs[file_extension]()
        except Exception as e:
            logger.error(f"Failed to write file: {file_location}. Error: {e}")
            raise

    return cleaned_df
