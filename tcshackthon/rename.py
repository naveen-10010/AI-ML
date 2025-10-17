import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Rename columns to match Streamlit app
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
df.rename(columns=column_rename_map, inplace=True)

# Encode categorical columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Normalize selected numerical columns
scaler = StandardScaler()
num_cols = ['Credit_amount', 'Duration', 'Age']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Prepare data
X = df.drop("Class", axis=1)
y = df["Class"].apply(lambda x: 0 if x == 2 else 1)  # 1 = Good, 0 = Bad

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the model
joblib.dump(rf, "credit_model.pkl")
print("✅ Model saved as 'credit_model.pkl'")

# Evaluate
y_pred = rf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance plot
importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.sort_values(ascending=False)[:20], y=importances.sort_values(ascending=False)[:20].index)
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.show()
