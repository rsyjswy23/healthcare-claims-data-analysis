from flask import Flask, request, jsonify
import joblib
import pandas as pd
# Add at the top of app.py
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

# Load trained model
model = joblib.load('model/claim_model.pkl')

# For encoding Diagnosis_Code, load mapping from training
df = pd.read_csv('../data/synthetic_health_claims.csv')
diagnosis_code_mapping = {code: idx for idx, code in enumerate(df['Diagnosis_Code'].astype('category').cat.categories)}

def preprocess_input(data):
    # Map Diagnosis_Code to category code
    diagnosis_code = data.get('diagnosisCode', '')
    diagnosis_code_encoded = diagnosis_code_mapping.get(diagnosis_code, -1)
    # Prepare DataFrame for prediction
    input_df = pd.DataFrame([{
        'Claim_Amount': float(data.get('claimAmount', 0)),
        'Patient_Age': int(data.get('patientAge', 0)),
        'Diagnosis_Code': diagnosis_code_encoded,
        'Healthcare_Provider': data.get('healthcareProvider', ''),
        'Provider_Specialty': data.get('providerSpecialty', ''),
        'Procedure_Code': data.get('procedureCode', ''),
        'Number_of_Procedures': int(data.get('numProcedures', 0)),
        'Hospital': data.get('hospital', '')
    }])
    # Only keep columns used by the model
    model_features = ['Claim_Amount', 'Patient_Age', 'Diagnosis_Code']
    input_df = input_df[model_features]
    return input_df

@app.route('/')
def home():
    return "Healthcare Claim Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = preprocess_input(data)
    prob = model.predict_proba(input_df)[0][1]
    return jsonify({'approval_likelihood': round(prob, 3)})

if __name__ == '__main__':
    app.run(debug=True)