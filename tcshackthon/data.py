import pandas as pd
df = pd.read_csv("german.data", sep=' ', header=None)
df.columns = [
    'Status', 'Duration_in_month', 'Credit_history', 'Purpose', 'Credit_amount',
    'Savings_account', 'Employment', 'Installment_rate', 'Personal_status_sex',
    'Debtors', 'Residence_since', 'Property', 'Age_in_years', 'Other_installment_plans',
    'Housing', 'Number_credits', 'Job', 'People_liable', 'Telephone', 'Foreign_worker', 'Class'
]
df.to_csv("german_credit_data.csv", index=False)
print("done")
