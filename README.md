# Automobile Loan Default Prediction

## Project Overview
This project focuses on predicting loan defaults for automobile loans using a machine learning pipeline. The dataset used in this project is highly imbalanced, making it a challenging problem. To address this issue, we applied resampling techniques such as **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset. A neural network model was developed to classify loan defaults, with metrics like precision, recall, and F1-score used to evaluate performance.

---

## Dataset
The dataset contains records of automobile loan applications, including various features like:
- Client income
- Loan annuity
- Credit amount
- Employment days
- Population density in the region

The target variable is `Default`, which indicates whether the client defaulted on their loan:
- `0`: No default
- `1`: Default

**Dataset Source**: [Kaggle - Automobile Loan Default Dataset](https://www.kaggle.com/competitions/automobile-loan-default-prediction/data)

---

## Methodology
1. **Data Cleaning and Preprocessing**:
   - Cleaned missing and inconsistent values using a custom data cleaning pipeline.
   - Applied feature scaling and one-hot encoding for numerical and categorical features.

2. **Handling Imbalanced Data**:
   - The dataset is highly imbalanced, with a majority of non-default cases (`0`).
   - Used **SMOTE** to generate synthetic samples for the minority class (`1`), ensuring a more balanced dataset.

3. **Model Development**:
   - Built a **neural network model** with the following architecture:
     - Input layer with normalization
     - Three dense layers with ReLU activation and dropout for regularization
     - Output layer with a sigmoid activation for binary classification
   - Optimized using Adam optimizer and binary cross-entropy loss.

4. **Evaluation**:
   - Evaluated the model using the confusion matrix, F1-score, precision, and recall.
   - Results highlight the challenges of imbalanced data and the improvements gained through SMOTE.

---

## Results
### Confusion Matrix:


![image](https://github.com/user-attachments/assets/574c0645-97c1-4079-be53-ceb7b2038a1d)


### Metrics:
- **F1-Score**: 0.23
- **Precision (Default)**: 81.5%
- **Recall (Default)**: 13.4%

The model demonstrates strong performance for the majority class but struggles with recall for the minority class. SMOTE helps improve recall but further optimization is required.

---

## How to Clone and Run
1. Clone the repository:
   ```bash
   git clone https://github.com/nicole-baltodano/automobile-loan-default-prediction.git
   cd automobile-loan-default-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```


