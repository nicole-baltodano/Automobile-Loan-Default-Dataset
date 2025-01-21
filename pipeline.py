from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_preprocessing_pipeline():
    """Create preprocessing pipeline for numeric and categorical features."""
    num_preproc = Pipeline([
        ("num_imputer", SimpleImputer(strategy="constant", fill_value=0.)),
        ("scaler", StandardScaler())
    ])

    cat_preproc = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"))
    ])

    return ColumnTransformer([
        ("num_tr", num_preproc, make_column_selector(dtype_include=["float64", "int64"])),
        ("cat_tr", cat_preproc, make_column_selector(dtype_include=["object"]))
    ])
