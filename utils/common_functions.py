import pandas as pd
from typing import Union
import pickle
import os


def read_dataframe_file(path_to_file: str) -> Union[pd.DataFrame, None]:
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    elif path_to_file.endswith('parquet'):
        return pd.read_parquet(path_to_file)
    else:
        raise ValueError("Unsupported file format")


def write_file(file, path):
    extension = os.path.splitext(path)[1]
    if extension == '.pickle':
        with open(path, 'wb') as f:
            pickle.dump(file, f)


def read_file(path):
    extension = os.path.splitext(path)[1]
    try:
        if extension == '.pickle':
            with open(path, 'rb') as f:
                file = pickle.load(f)
        else:
            print('Unknown extension')
            return None
    except FileNotFoundError:
        print('File not found')
        return None
    return file
