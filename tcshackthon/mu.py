# german_credit_multi_model_clean.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# 1. Load and Convert Raw Data (space-separated)
df = pd.read_csv("german.data", sep=' ', header=None)
df.columns = [
    'Status', 'Duration_in_month', 'Credit_history', 'Purpose', 'Credit_amount',
    'Savings_account', 'Employment', 'Installment_rate', 'Personal_status_sex',
    'Debtors', 'Residence_since', 'Property', 'Age_in_years', 'Other_installment_plans',
    'Housing', 'Number_credits', 'Job', 'People_liable', 'Telephone', 'Foreign_worker', 'Class'
]

# 2. Encode categorical features
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 3. Normalize numerical features
scaler = StandardScaler()
numeric_cols = ['Credit_amount', 'Duration_in_month', 'Age_in_years']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 4. Define features and target
X = df.drop("Class", axis=1)
y = df["Class"].apply(lambda x: 1 if x == 1 else 0)  # 1 = Good, 0 = Bad

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', verbosity=0),  # Removed deprecated param
    "LightGBM": LGBMClassifier(force_col_wise=True),  # Avoid row-wise overhead warning
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        max_iter=1000, 
        early_stopping=True, 
        random_state=42
    )
}

# 7. Fit, Predict, and Evaluate
for name, model in models.items():
    print(f"\n🔷 {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
