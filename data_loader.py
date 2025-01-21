import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path: str):
    """Load the dataset."""
    data = pd.read_csv(file_path)
    return data

def split_data(data, target_column="Default", test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = data.drop(columns=[target_column, "ID"])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
