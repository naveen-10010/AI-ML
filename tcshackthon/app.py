import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from io import BytesIO

# Load the trained model
model = joblib.load("credit_model.pkl")

# Mapping original dataset columns to those used during training
column_rename_map = {
    "Status_of_existing_checking_account": "Checking_status",
    "Duration_in_month": "Duration",
    "Credit_history": "Credit_history",
    "Purpose": "Purpose",
    "Credit_amount": "Credit_amount",
    "Savings_account/bonds": "Savings",
    "Present_employment_since": "Employment",
    "Installment_rate_in_percentage_of_disposable_income": "Installment_rate",
    "Personal_status_and_sex": "Personal_status",
    "Other_debtors_or_guarantors": "Debtors",
    "Present_residence_since": "Residence_since",
    "Property": "Property",
    "Age_in_years": "Age",
    "Other_installment_plans": "Other_installment_plans",
    "Housing": "Housing",
    "Number_of_existing_credits_at_this_bank": "Number_credits",
    "Job": "Job",
    "Number_of_people_being_liable_to_provide_maintenance_for": "People_liable",
    "Telephone": "Telephone",
    "foreign_worker": "Foreign_worker"
}

# Data preprocessing function
def preprocess_data(df):
    # Rename columns
    df.rename(columns=column_rename_map, inplace=True)

    # Label encode categorical columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Scale numerical columns
    num_cols = ['Credit_amount', 'Duration', 'Age']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

# Streamlit UI
st.title("Credit Risk Prediction App")

option = st.radio("Choose Prediction Mode:", ["Single Person", "Bulk"])

if option == "Single Person":
    st.subheader("Enter Applicant Details")

    user_input = {
        "Status_of_existing_checking_account": st.selectbox("Checking Account Status", ["<0", "0<=X<200", ">=200", "no checking"]),
        "Duration_in_month": st.number_input("Duration (months)", min_value=6, max_value=72, value=24),
        "Credit_history": st.selectbox("Credit History", ["no credits", "all paid", "existing paid", "critical", "delayed"]),
        "Purpose": st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "education", "business", "others"]),
        "Credit_amount": st.number_input("Credit Amount", min_value=250, max_value=20000, value=5000),
        "Savings_account/bonds": st.selectbox("Savings Account", ["<100", "100<=X<500", "500<=X<1000", ">=1000", "unknown"]),
        "Present_employment_since": st.selectbox("Employment Since", ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]),
        "Installment_rate_in_percentage_of_disposable_income": st.selectbox("Installment Rate (%)", [1, 2, 3, 4]),
        "Personal_status_and_sex": st.selectbox("Personal Status & Sex", ["male single", "female div/dep/mar", "male div/sep", "male mar/wid"]),
        "Other_debtors_or_guarantors": st.selectbox("Other Debtors/Guarantors", ["none", "co-applicant", "guarantor"]),
        "Present_residence_since": st.selectbox("Present Residence Since", [1, 2, 3, 4]),
        "Property": st.selectbox("Property", ["real estate", "life insurance", "car", "no known property"]),
        "Age_in_years": st.number_input("Age", min_value=18, max_value=75, value=30),
        "Other_installment_plans": st.selectbox("Other Installment Plans", ["none", "bank", "stores"]),
        "Housing": st.selectbox("Housing", ["own", "for free", "rent"]),
        "Number_of_existing_credits_at_this_bank": st.selectbox("Number of Existing Credits", [1, 2, 3, 4]),
        "Job": st.selectbox("Job", ["unemployed/unskilled non-res", "unskilled resident", "skilled", "highly skilled"]),
        "Number_of_people_being_liable_to_provide_maintenance_for": st.selectbox("Number Liable", [1, 2]),
        "Telephone": st.selectbox("Telephone", ["none", "yes"]),
        "foreign_worker": st.selectbox("Foreign Worker", ["yes", "no"])
    }

    input_df = pd.DataFrame([user_input])
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

            # Excel download option
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
            st.error(f"Error during processing: {e}")
