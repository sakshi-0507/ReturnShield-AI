# ReturnShield AI

### E-commerce Return & Refund Abuse Detection

## Overview

ReturnShield AI is a machine learning project that classifies e-commerce return records into four categories: Legitimate, Policy Abuser, Fraudulent Return, and Wardrobing.

The project uses a Random Forest classifier and a Streamlit web application to demonstrate how machine learning can support return-risk screening.

## Problem Statement

E-commerce businesses need ways to identify potentially abusive return patterns while avoiding unnecessary scrutiny of legitimate customers. This project explores classification using order, return, and customer-behavior features.

## Features

* Data exploration and preparation
* Missing-value imputation and categorical encoding
* Multiclass classification using Random Forest
* Model evaluation with accuracy, precision, recall, F1-score, and a confusion matrix
* Feature-importance analysis
* Interactive Streamlit prediction interface

## Target Categories

* Legitimate
* Policy Abuser
* Fraudulent Return
* Wardrobing

## Technology Stack

* Python
* Pandas and NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

## Model Evaluation

The current Random Forest model achieved 99.92% accuracy on a held-out test split of the dataset used during development.

**Important limitation:** The dataset is synthetic. This result does not establish performance on real-world transactions. Further validation is required before production use.

## Run Locally

1. Install the required Python packages.
2. Ensure the saved model and preprocessor files are in the `returnshield_models` folder.
3. Run the application:

```bash
python -m streamlit run app.py
```

4. Open the local URL displayed in your terminal.

## Project Structure

```text
Projects_ML/
├── app.py
├── README.md
└── returnshield_models/
    ├── random_forest_model.pkl
    └── preprocessor.pkl
```

## Responsible Use

Predictions are decision-support signals, not proof of fraud. Any potentially abusive return should receive appropriate human review. The model has not been validated for real-world deployment.

## Future Improvements

* Validate on real or independently generated data
* Investigate potential label-generation shortcuts
* Compare Random Forest with other classification models
* Add automated tests and clearer risk explanations
* Deploy the application after suitable validation

## Author

Sakshi Bagul
