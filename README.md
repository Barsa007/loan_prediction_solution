# 🏦 Loan Approval Prediction App

A Machine Learning project that predicts whether a loan application will be approved or denied,  
based on applicant financial and personal data.

The project includes:
- Model training and evaluation
- A trained ML model saved to disk
- An interactive Streamlit web application for predictions
- Logging and model performance visualization

---

## 📌 Project Overview

This project trains a **Support Vector Machine (SVM)** classifier to predict loan approval status  
(`Approved / Denied`) using historical loan application data.

The trained model is then deployed in a **Streamlit** application that allows users to:
- Enter loan application details
- Get an instant prediction
- View model performance metrics
- Log predictions and system events

---

## 🧠 Machine Learning Pipeline

- Data preprocessing (missing values, encoding)
- Feature scaling using `StandardScaler`
- Model training with **SVM**
- Hyperparameter tuning using `GridSearchCV`
- Model evaluation on a holdout test set
- Saving model + metadata using `joblib`

---

## 📊 Model Performance

Model performance is evaluated during training and saved together with the model.

Metrics include:
- Accuracy
- Classification report (precision, recall, F1-score)

These metrics can be displayed directly from the Streamlit app via a dedicated button.

---

## 🖥️ Streamlit Application Features

- User-friendly loan application form
- Real-time prediction (Approved / Denied)
- Model accuracy display
- Logging of:
  - App startup
  - User prediction requests
  - Model prediction results
  - Errors (if any)

---

## 📂 Project Structure

```text
.
├── train.py                     # Model training and evaluation script
├── main.py                      # Streamlit application
├── loan_shark_with_metrics.joblib  # Trained model bundle
├── app.log                      # Application logs
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
├── sample_submission.csv        # Submission format example
├── README.md                    # Project documentation
