#  Fraud Detection & Risk Analytics System

A machine learning-based web application that detects potentially fraudulent financial transactions and provides real-time risk analysis.

The system uses a Random Forest machine learning model to classify transactions as **Genuine** or **Fraudulent** and calculates a risk score and risk level.

---

##  Project Overview

Fraud Detection & Risk Analytics is a Flask-based machine learning web application designed to analyze financial transactions and identify suspicious activity.

The application allows users to:

- Register and login securely
- Analyze individual transactions
- Predict whether a transaction is fraudulent
- Calculate fraud risk score
- Classify transactions into Low, Medium, and High risk
- Store transaction history
- Search and filter transactions
- Delete transaction records
- Upload CSV files for bulk fraud analysis
- Download fraud analysis results
- View analytics through a dashboard

---

##  Machine Learning Model

The project uses a **Random Forest Classifier** for fraud detection.

### Model Features

The model analyzes the following transaction features:

- Transaction Amount
- Account Age
- Previous Transactions
- Failed Transactions
- Transaction Hour
- International Transaction
- Device Change

### Prediction

The machine learning model generates:

```text
Genuine Transaction
or
Fraudulent Transaction
