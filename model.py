from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def create_model_pipeline(preproc_pipeline, type_cleaner):
    """Create the model pipeline with SMOTE and calibrated classifier."""
    return ImbPipeline([
        ("cleaner", type_cleaner),
        ("preprocessing", preproc_pipeline),
        ("smote", SMOTE(random_state=42)),
        ("classifier", CalibratedClassifierCV(RandomForestClassifier(class_weight="balanced"), cv=5))
    ])
