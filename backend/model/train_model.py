import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your data
df = pd.read_csv('../../data/synthetic_health_claims.csv')

# Simple preprocessing (example: select relevant columns)
features = ['Claim_Amount', 'Patient_Age', 'Diagnosis_Code']  # Match frontend fields
df = df.dropna(subset=features + ['Is_Fraudulent'])  # Ensure no missing values

# Encode categorical feature
df['Diagnosis_Code'] = df['Diagnosis_Code'].astype('category').cat.codes

X = df[features]
y = df['Is_Fraudulent']  # 1 for approved, 0 for not approved

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'claim_model.pkl')
print("Model trained and saved to model/claim_model.pkl")