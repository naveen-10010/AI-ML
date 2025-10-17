import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv("german_credit_data.csv")
df.head()
df.info()
df.describe()
df['Class'].value_counts()
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
scaler = StandardScaler()
num_cols = ['Credit_amount', 'Duration_in_month', 'Age_in_years']
df[num_cols] = scaler.fit_transform(df[num_cols])
X = df.drop('Class', axis=1)
y = df['Class'].apply(lambda x: 0 if x == 2 else 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
