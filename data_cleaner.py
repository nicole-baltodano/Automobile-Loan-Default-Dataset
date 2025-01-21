import pandas as pd
import numpy as np
import string
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer

# Cleaner function
def clean_df(X):
    def clean_value(value):
        unwanted_char = string.punctuation.replace(".", "") + string.ascii_letters
        for tag in unwanted_char:
            value = value.replace(tag, "")
        value = np.nan if not value else value
        return float(value)

    X = pd.DataFrame(X)
    X = X.astype(str).applymap(clean_value)
    return X

# Column transformer
def create_type_cleaner(columns_to_fix):
    cleaner = FunctionTransformer(clean_df)
    return ColumnTransformer(
        [("type_cleaner", cleaner, columns_to_fix)],
        remainder="passthrough"
    ).set_output(transform="pandas")
