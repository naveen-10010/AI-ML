import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from io import BytesIO

# Load model
model = joblib.load("credit_model.pkl")

# Define categorical columns for label encoding
categorical_columns = [
    "Status_of_existing_checking_account",
    "Credit_history",
    "Purpose",
    "Savings_account_bonds",
    "Present_employment_since",
    "Personal_status_and_sex",
    "Other_debtors_or_guarantors",
    "Property",
    "Other_installment_plans",
    "Housing",
    "Job",
    "Telephone",
    "Foreign_worker"
]

# Numerical columns to scale
numerical_columns = [
    "Duration_in_month",
    "Credit_amount",
    "Installment_rate_in_percentage_of_disposable_income",
    "Present_residence_since",
    "Age_in_years",
    "Number_of_existing_credits_at_this_bank",
    "Number_of_people_being_liable_to_provide_maintenance_for"
]

# Preprocess input data
le_dict = {col: LabelEncoder() for col in categorical_columns}
scaler = StandardScaler()

def preprocess_data(df):
    for col in categorical_columns:
        df[col] = le_dict[col].fit_transform(df[col])
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

# Streamlit UI
st.title("Credit Risk Prediction App")
option = st.radio("Choose Prediction Mode:", ["Single Person", "Bulk"])

if option == "Single Person":
    st.subheader("Enter Applicant Details")
    input_dict = {
        "Status_of_existing_checking_account": st.selectbox("Status of Checking Account", ["<0 DM", "0<=X<200 DM", ">=200 DM", "no checking account"]),
        "Duration_in_month": st.number_input("Loan Duration (Months)", 6, 72, 24),
        "Credit_history": st.selectbox("Credit History", ["no credits taken", "all paid", "existing paid", "delayed", "critical account"]),
        "Purpose": st.selectbox("Purpose", ["car", "furniture", "radio/TV", "education", "business", "repairs", "vacation", "others"]),
        "Credit_amount": st.number_input("Credit Amount", 100, 100000, 5000),
        "Savings_account_bonds": st.selectbox("Savings Account/Bonds", ["<100 DM", "100<=X<500 DM", "500<=X<1000 DM", ">=1000 DM", "unknown"]),
        "Present_employment_since": st.selectbox("Employment Since", ["unemployed", "<1 year", "1<=X<4 years", "4<=X<7 years", ">=7 years"]),
        "Installment_rate_in_percentage_of_disposable_income": st.slider("Installment Rate %", 1, 4, 2),
        "Personal_status_and_sex": st.selectbox("Personal Status & Sex", ["male single", "female div/dep/mar", "male div/sep", "male mar/wid"]),
        "Other_debtors_or_guarantors": st.selectbox("Other Debtors/Guarantors", ["none", "co-applicant", "guarantor"]),
        "Present_residence_since": st.slider("Present Residence Since (Years)", 1, 4, 2),
        "Property": st.selectbox("Property", ["real estate", "building society savings", "car", "unknown"]),
        "Age_in_years": st.number_input("Age", 18, 100, 30),
        "Other_installment_plans": st.selectbox("Other Installment Plans", ["bank", "stores", "none"]),
        "Housing": st.selectbox("Housing", ["own", "for free", "rent"]),
        "Number_of_existing_credits_at_this_bank": st.slider("No. of Existing Credits", 1, 4, 1),
        "Job": st.selectbox("Job", ["unemployed", "unskilled", "skilled", "highly skilled"]),
        "Number_of_people_being_liable_to_provide_maintenance_for": st.slider("No. of Liable People", 1, 2, 1),
        "Telephone": st.selectbox("Telephone", ["none", "yes"]),
        "Foreign_worker": st.selectbox("Foreign Worker", ["yes", "no"])
    }

    input_df = pd.DataFrame([input_dict])
    processed_input = preprocess_data(input_df)
    prediction = model.predict(processed_input)[0]
    st.success("Prediction: " + ("Good Credit" if prediction == 1 else "Bad Credit"))

elif option == "Bulk":
    st.subheader("Upload Excel File for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df_original = df.copy()
        try:
            df_processed = preprocess_data(df)
            predictions = model.predict(df_processed)
            df_original["Prediction"] = ["Good Credit" if p == 1 else "Bad Credit" for p in predictions]
            st.success("Predictions complete!")
            st.dataframe(df_original)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_original.to_excel(writer, index=False, sheet_name='Predictions')
            st.download_button(
                label="Download Predictions Excel",
                data=output.getvalue(),
                file_name="credit_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error while processing: {e}")
