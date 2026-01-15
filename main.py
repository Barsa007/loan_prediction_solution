"""
Loan Approval Checker (Streamlit)

Loads a trained loan-approval model bundle (model + feature columns + metrics)
and provides a UI for predicting whether a loan will be approved.

Run:
    streamlit run main.py
"""

import logging
import os
from typing import Dict, Any

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)


MODEL_PATH = "loan_shark_with_metrics.joblib"
LOG_PATH = "app.log"


def setup_logging(log_path: str = LOG_PATH) -> None:
    """
    Configure application logging to a file.

    Args:
        log_path (str): Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        filename=log_path
    )
    logging.info("Application started")


@st.cache_resource
def load_bundle(path: str) -> Dict[str, Any]:
    """
    Load the saved model bundle from disk.

    Expected keys in the bundle:
        - "model": trained sklearn estimator/pipeline
        - "feature_cols": pandas Index / list of feature column names
        - "metrics": dict with evaluation metrics (optional but recommended)

    Args:
        path (str): Path to the joblib file.

    Returns:
        dict: The loaded bundle.
    """
    return joblib.load(path)


def ensure_model_file_exists(path: str) -> None:
    """
    Stop the app gracefully if the model file is missing.

    Args:
        path (str): Path to the model file.
    """
    if not os.path.exists(path):
        st.warning("The system is initializing, please wait.")
        st.error("Model file not found. Train the model and save it before running the app.")
        logging.error("Model file not found at path: %s", os.path.abspath(path))
        st.stop()


def build_input_dataframe(
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_term: float,
    credit_history: int,
    married: str,
) -> pd.DataFrame:
    """
    Build a raw input DataFrame from user inputs (before encoding).

    Note:
        We keep 'Married' as Yes/No (string) so get_dummies matches training style.

    Returns:
        pd.DataFrame: Single-row dataframe with raw features.
    """
    return pd.DataFrame([{
        "ApplicantIncome": float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": int(credit_history),
        "Married": "Yes" if married == "Yes" else "No",
    }])


def preprocess_for_model(df_raw: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """
    Apply the same preprocessing logic used in training (basic dummy encoding + NA fill),
    then align columns to the training feature set.

    Args:
        df_raw (pd.DataFrame): Raw input dataframe.
        feature_cols: Feature columns from training (Index or list).

    Returns:
        pd.DataFrame: Model-ready dataframe with aligned columns.
    """
    df = pd.get_dummies(df_raw, drop_first=True).fillna(0)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


def render_model_performance(bundle: Dict[str, Any]) -> None:
    """
    Render model performance metrics (if available) in the UI.
    """
    if "metrics" not in bundle:
        st.error("No metrics found in the saved model bundle. Re-train and save metrics.")
        logging.warning("User requested metrics, but 'metrics' key is missing from bundle.")
        return

    acc = float(bundle["metrics"].get("accuracy", 0.0))
    st.metric("Model Accuracy", f"{acc * 100:.2f}%")

    report = bundle["metrics"].get("report")
    if report is not None:
        with st.expander("Show classification report (JSON)"):
            st.json(report)


def predict_loan(model, X: pd.DataFrame) -> str:
    """
    Predict loan approval status using the trained model.

    Args:
        model: Trained sklearn estimator/pipeline.
        X (pd.DataFrame): Model-ready input dataframe.

    Returns:
        str: 'Y' for approved, 'N' for denied.
    """
    pred = model.predict(X)[0]
    return str(pred)


def main() -> None:
    """
    Main Streamlit application entry point.
    """
    st.set_page_config(page_title="Loan Approval Checker", page_icon="🏦", layout="centered")
    st.title("🏦 Loan Approval Checker")

    setup_logging()
    ensure_model_file_exists(MODEL_PATH)

    # Load bundle (cached)
    bundle = load_bundle(MODEL_PATH)

    # Validate bundle keys
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_cols" not in bundle:
        st.error("Invalid model bundle format. Expected keys: 'model', 'feature_cols' (and optionally 'metrics').")
        logging.error("Invalid bundle format loaded from %s. Type: %s", MODEL_PATH, type(bundle))
        st.stop()

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Button: model performance
    if st.button("📊 Show Model Performance"):
        render_model_performance(bundle)

    # Input form
    with st.form("loan_form"):
        st.subheader("Loan Application Details")

        applicant_income = st.number_input("Applicant Income (monthly)", min_value=0.0, value=5000.0, step=100.0)
        coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0.0, value=0.0, step=100.0)
        loan_amount = st.number_input("Requested Loan Amount", min_value=0.0, value=150.0, step=10.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1.0, value=360.0, step=12.0)

        credit_history = st.selectbox(
            "Credit History",
            options=[1, 0],
            index=0,
            format_func=lambda x: "Exists (1)" if x == 1 else "Does not exist (0)"
        )

        dict_opt = {"Yes": "Married", "No": "Single"}
        married = st.selectbox(
            "Marital Status",
            options=["Yes", "No"],
            index=0,
            format_func=lambda x: dict_opt[x]
        )

        submitted = st.form_submit_button("Check Loan Eligibility")

    if submitted:
        # Build + preprocess
        df_raw = build_input_dataframe(
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            loan_term=loan_term,
            credit_history=credit_history,
            married=married,
        )

        X = preprocess_for_model(df_raw, feature_cols)

        logging.info("Prediction requested")
        logging.info("Input raw: %s", df_raw.to_dict(orient="records"))
        logging.info("Input aligned columns count: %d", X.shape[1])

        try:
            prediction = predict_loan(model, X)
            logging.info("Prediction result: %s", prediction)

            if prediction == "Y":
                st.success("Loan Application Approved!")
            elif prediction == "N":
                st.error("Loan Application Denied")
            else:
                st.warning(f"Unexpected prediction value: {prediction}")

        except Exception as e:
            logging.exception("Prediction failed")
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()