# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC

# ------------------ UI ------------------
st.title("AutoML Web App")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Dataset", df.head())

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    missing_option = st.selectbox("Handle missing values by", ["Drop Rows", "Fill Mean", "Fill Median"])
    if missing_option == "Drop Rows":
        df = df.dropna()
    elif missing_option == "Fill Mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_option == "Fill Median":
        df = df.fillna(df.median(numeric_only=True))

    # Select Target and Features
    target = st.selectbox("Select target variable", df.columns)

    # Select All Option
    all_columns_except_target = [col for col in df.columns if col != target]
    select_all = st.checkbox("Select all features")

    if select_all:
        features = st.multiselect("Select feature variables", all_columns_except_target, default=all_columns_except_target)
    else:
        features = st.multiselect("Select feature variables", all_columns_except_target)

    # Validation
    if not features:
        st.warning("⚠️ Please select at least one feature to proceed.")
        st.stop()

    X = df[features]
    y = df[target]

    # Normalization
    scaling = st.selectbox("Choose normalization method", ["None", "Min-Max", "Z-Score"])
    if scaling == "Min-Max":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif scaling == "Z-Score":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Train/Test Split
    test_size = st.slider("Test set ratio", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Model Selection
    task = st.radio("Task Type", ["Regression", "Classification"])
    model_name = st.selectbox("Choose Model", ["Linear/Logistic Regression", "Random Forest", "SVM"])

    # Hyperparameters
    st.subheader("Hyperparameter Options")

    if model_name == "Linear/Logistic Regression":
        params = {"C": st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)}
    elif model_name == "Random Forest":
        params = {
            "n_estimators": st.slider("Number of trees", 10, 200, 100),
            "max_depth": st.slider("Max Depth", 1, 20, 5)
        }
    elif model_name == "SVM":
        params = {
            "C": st.slider("Regularization (C)", 0.01, 10.0, 1.0),
            "kernel": st.selectbox("Kernel", ["linear", "rbf", "poly"])
        }

    # Train Button
    if st.button("Train Model"):
        if task == "Regression":
            if model_name == "Linear/Logistic Regression":
                model = LinearRegression()
            elif model_name == "Random Forest":
                model = RandomForestRegressor(**params)
            elif model_name == "SVM":
                model = SVR(**params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Regression Metrics")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
            st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
            st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

        else:  # Classification
            if model_name == "Linear/Logistic Regression":
                model = LogisticRegression(C=params['C'], max_iter=1000)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(**params)
            elif model_name == "SVM":
                model = SVC(**params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Classification Metrics")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            st.write(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
            st.write(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
            st.write(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")