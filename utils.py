import pandas as pd
import csv



def prepare_dataset(df, target_col):
    """
    Removes rows with missing target values and cleaned dataframe, features and target
    """
    missing_rows = df[target_col].isna().sum()
    if missing_rows > 0:
         df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return df, X, y, missing_rows

def detect_problem_type(y):
    """
    Detects the type of machine learning problem based on the target variable.
    """ 
    if y.dtype == "object" or y.nunique() < 10:
        return "Classification"
    return "Regression"

def load_csv(uploaded_file):
    """
    Loads a CSV file into a pandas DataFrame.
    """
    sample = uploaded_file.getvalue().decode("utf-8")

    dialect = csv.Sniffer().sniff(
         sample,
         delimiters=[",", ";", "\t"])
    
    df = pd.read_csv(
        uploaded_file, 
        sep=dialect.delimiter)
    
    return df