from data_loader import load_data, split_data
from data_cleaner import create_type_cleaner
from pipeline import create_preprocessing_pipeline
from model import create_model_pipeline
from evaluation import evaluate_model

def main():
    # Load data
    data = load_data("data/Train_Dataset.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # Create type cleaner and preprocessing pipeline
    columns_to_fix = ["Score_Source_2", "Score_Source_3", "Client_Income", "Credit_Amount",
                      "Loan_Annuity", "Age_Days", "Employed_Days", "Registration_Days",
                      "ID_Days", "Population_Region_Relative"]
    type_cleaner = create_type_cleaner(columns_to_fix)
    preproc_pipeline = create_preprocessing_pipeline()

    # Create model pipeline
    pipe = create_model_pipeline(preproc_pipeline, type_cleaner)

    # Train model
    pipe.fit(X_train, y_train)

    # Evaluate model
    evaluate_model(pipe, X_test, y_test)

if __name__ == "__main__":
    main()
