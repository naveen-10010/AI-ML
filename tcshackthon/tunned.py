import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset and clean
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")  # Replace with your dataset path
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Prepare data
X = df.drop("Risk", axis=1)
y = df["Risk"]

# One-hot encoding for categorical
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
@st.cache_resource
def train_model():
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model

model = train_model()

# Evaluate model
scores = cross_val_score(model, X_train, y_train, cv=5)
st.sidebar.success(f"Model Accuracy (CV): {np.mean(scores):.2f}")

# UI
st.title("Credit Risk Prediction App")

option = st.radio("Choose Prediction Mode", ["Single Person", "Bulk (Excel Upload)"])

def make_prediction(input_df):
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    return model.predict(input_df)

if option == "Single Person":
    st.subheader("Enter Details")

    with st.form("person_form"):
        age = st.slider("Age", 18, 100, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job (0=unskilled, 3=highly skilled)", [0, 1, 2, 3])
        housing = st.selectbox("Housing", ["own", "free", "rent"])
        saving_account = st.selectbox("Saving Account", ["little", "moderate", "quite rich", "rich", np.nan])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", np.nan])
        credit_amount = st.number_input("Credit Amount", min_value=0)
        duration = st.number_input("Duration (Months)", min_value=1)
        purpose = st.selectbox("Purpose", ['radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'domestic appliances', 'repairs', 'vacation/others'])

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame({
                "Age": [age],
                "Sex": [sex],
                "Job": [job],
                "Housing": [housing],
                "Saving accounts": [saving_account],
                "Checking account": [checking_account],
                "Credit amount": [credit_amount],
                "Duration": [duration],
                "Purpose": [purpose]
            })
            prediction = make_prediction(input_data)
            st.success(f"Prediction: {prediction[0]}")

elif option == "Bulk (Excel Upload)":
    st.subheader("Upload Excel File")

    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            bulk_data = pd.read_excel(uploaded_file)
            if "Name" not in bulk_data.columns:
                st.error("Excel must contain a 'Name' column.")
            else:
                input_data = bulk_data.drop("Name", axis=1)
                prediction = make_prediction(input_data)
                bulk_data["Prediction"] = prediction

                st.success("Predictions completed.")
                st.dataframe(bulk_data[["Name", "Prediction"]])

                output_file = "credit_risk_predictions.xlsx"
                bulk_data.to_excel(output_file, index=False)

                with open(output_file, "rb") as f:
                    st.download_button("Download Prediction File", f, file_name="predictions.xlsx")

        except Exception as e:
            st.error(f"Error: {str(e)}")
