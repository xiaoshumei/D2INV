import os
import re
import pandas as pd
import logging

logger = logging.getLogger("D2INV")

data_dir = "../datasets/"


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
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).round(2)
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna(subset=numerical_columns)
    return cleaned_df


def preprocess_response(response):
    if "```" in response:
        pattern = r"```(?:\w+\n)?([\s\S]+?)```"
        matches = re.findall(pattern, response)
        if matches:
            response = matches[0]
    return response


def filter_dataframe(data_summary: dict, df: pd.DataFrame) -> str:
    """
    Filter the dataframe by dropping the nan values.
    :param data_summary: dataset summary
    :param df: data frame to be filtered
    :return: data frame after filter
    """
    numerical_columns = [
        column["column"]
        for column in data_summary["fields"]
        if column["properties"]["dtype"] == "number"
    ]
    df = clean_nan_rows(df, numerical_columns)
    data = df.to_json(orient="records", force_ascii=False)
    return data


def read_dataframe(
    file_name, encoding: str = "utf-8", max_rows=5000
) -> [pd.DataFrame, pd.DataFrame]:
    """
    Read a dataframe from a given file location and clean its column names.
    It also samples down to 4500 rows if the data exceeds that limit.

    :param file_name: The name of the data file.
    :param encoding: Encoding to use for the file reading.
    :param max_rows: Max rows to split.
    :return: A cleaned DataFrame and a split DataFrame.
    """
    file_location = os.path.join(data_dir, file_name)
    file_extension = file_location.split(".")[-1]

    read_funcs = {
        "json": lambda: pd.read_json(
            file_location, orient="records", encoding=encoding, convert_dates=False
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
    split_df = cleaned_df.copy()
    # Sample down to limit rows if necessary
    if len(split_df) > max_rows:
        logger.info(
            f"Dataframe has more than {max_rows} rows. We will sample 4500 rows."
        )
        split_df = split_df.sample(max_rows)

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

    return [cleaned_df, split_df]
