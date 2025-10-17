import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load and preprocess data
df = pd.read_csv("german_credit_data.csv")
cat_cols = df.select_dtypes(include='object').columns

# Encode categorical features
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Scale numerical features
scaler = StandardScaler()
num_cols = ['Credit_amount', 'Duration_in_month', 'Age_in_years']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Prepare features and target
X = df.drop('Class', axis=1)
y = df['Class'].apply(lambda x: 0 if x == 2 else 1)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aggressive parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Evaluate best model
y_pred = best_rf.predict(X_test)
print("\n✅ Best Parameters:", grid_search.best_params_)
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("🔍 Precision:", precision_score(y_test, y_pred))
print("📈 Recall:", recall_score(y_test, y_pred))
print("📊 F1 Score:", f1_score(y_test, y_pred))

# Compare with default model using cross-validation
default_rf = RandomForestClassifier(random_state=42)
cv_default = cross_val_score(default_rf, X, y, cv=5)
cv_optimized = cross_val_score(best_rf, X, y, cv=5)

print("\n📉 Default RF CV Accuracy:", np.mean(cv_default))
print("📈 Optimized RF CV Accuracy:", np.mean(cv_optimized))

# Save the best model
try:
    joblib.dump(best_rf, "credit_model_tuned.pkl")
    print("💾 Tuned model saved successfully.")
except Exception as e:
    print("❌ Error saving model:", e)

# Feature importance
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh')
plt.title("Top 15 Important Features")
plt.tight_layout()
plt.show()
