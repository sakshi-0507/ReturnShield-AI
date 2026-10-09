
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="ReturnShield AI",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_models():
    model = joblib.load(
        "returnshield_models/random_forest_model.pkl"
    )
    preprocessor = joblib.load(
        "returnshield_models/preprocessor.pkl"
    )
    return model, preprocessor


st.title("ReturnShield AI")
st.subheader("E-commerce Return & Refund Abuse Detection")
st.write(
    "Enter order details to analyze a potential return-abuse category."
)

try:
    model, preprocessor = load_models()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()


numeric_cols = list(preprocessor.transformers_[0][2])
categorical_cols = list(preprocessor.transformers_[1][2])

numeric_imputer = (
    preprocessor.named_transformers_["num"]
    .named_steps["imputer"]
)

categorical_pipeline = preprocessor.named_transformers_["cat"]
categorical_imputer = categorical_pipeline.named_steps["imputer"]
encoder = categorical_pipeline.named_steps["encoder"]

numeric_defaults = dict(
    zip(numeric_cols, numeric_imputer.statistics_)
)

categorical_defaults = dict(
    zip(categorical_cols, categorical_imputer.statistics_)
)

category_options = dict(
    zip(categorical_cols, encoder.categories_)
)


st.divider()
st.header("Order Details")

st.caption(
    "Enter order details below. Default values are provided "
    "for testing and may not represent a real order."
)

input_data = {}

with st.form("return_prediction_form"):
    col1, col2 = st.columns(2)

    for i, feature in enumerate(preprocessor.feature_names_in_):
        container = col1 if i % 2 == 0 else col2

        with container:
            field_label = feature.replace("_", " ").title()

            if feature in numeric_defaults:
                default = numeric_defaults[feature]

                if pd.isna(default) or not np.isfinite(default):
                    default = 0.0

                input_data[feature] = st.number_input(
                    field_label,
                    value=float(default),
                    key=f"input_{feature}"
                )
            else:
                options = list(category_options[feature])

                if not options:
                    options = [categorical_defaults[feature]]

                default = categorical_defaults[feature]
                default_index = (
                    options.index(default)
                    if default in options else 0
                )

                input_data[feature] = st.selectbox(
                    field_label,
                    options=options,
                    index=default_index,
                    key=f"input_{feature}"
                )

    submitted = st.form_submit_button("Analyze Return")


if submitted:
    input_df = pd.DataFrame(
        [input_data],
        columns=preprocessor.feature_names_in_
    )

    try:
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)[0]

        labels = {
            0: "Legitimate",
            1: "Policy Abuser",
            2: "Fraudulent Return",
            3: "Wardrobing"
        }

        st.divider()
        st.header("Prediction Result")

        if prediction in labels:
            predicted_label = labels[prediction]

            if prediction == 0:
                st.success(
                    f"Predicted Category: {predicted_label}"
                )
                st.write(
                    "The model classified this return as legitimate. "
                    "This is a prediction, not a guarantee."
                )

            elif prediction == 1:
                st.warning(
                    f"Predicted Category: {predicted_label}"
                )
                st.write(
                    "The model flagged this return as potentially "
                    "involving policy abuse. Review the order details."
                )

            elif prediction == 2:
                st.error(
                    f"Predicted Category: {predicted_label}"
                )
                st.write(
                    "The model flagged this return as potentially "
                    "fraudulent. Further manual investigation is needed."
                )

            elif prediction == 3:
                st.warning(
                    f"Predicted Category: {predicted_label}"
                )
                st.write(
                    "The model flagged this return as potentially "
                    "involving wardrobing. Review the order details "
                    "before making a decision."
                )

            st.caption(f"Predicted class: {prediction}")

        else:
            st.info(
                "Unknown predicted class. Verify the dataset's "
                "label mapping."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")