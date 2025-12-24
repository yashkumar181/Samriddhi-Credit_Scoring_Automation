# scorer.py (Final Corrected Version)

import joblib
import pandas as pd
import numpy as np
import shap

# --- 1. LOAD FINAL MODELS AND ARTIFACTS ---
# ... (Keep existing loading block) ...
try:
    # Model A (Repayment) - trained on Loan_default data
    repayment_model = joblib.load('./saved_models/repayment_model_xgb.joblib')
    repayment_features = joblib.load('./saved_models/repayment_model_features.joblib')
    repayment_explainer = joblib.load('./saved_models/repayment_model_explainer.joblib')

    # Model B (Income) - trained on socio-economic data
    income_model = joblib.load('./saved_models/income_model_final.joblib')
    income_features = joblib.load('./saved_models/income_model_final_features.joblib')
    income_explainer = joblib.load('./saved_models/income_model_final_explainer.joblib')
except FileNotFoundError as e:
    print(f"FATAL ERROR: A required model file was not found: {e}")
    repayment_model = income_model = None


# --- 2. DEFINE CATEGORICAL FEATURES for BOTH Models ---
MODEL_A_CAT_FEATURES = [
    'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
    'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

MODEL_B_CAT_FEATURES = [
    'Sector', 'Social_Group_of_HH_Head', 'Max_Income_Activity', 
    'Type_of_Dwelling_Unit', 'Land_Ownership', 'Ration_Card_Type',
    'Religion_of_HH_Head'
]

# --- 3. MAIN SCORING AND EXPLAINING FUNCTIONS ---
def get_prepared_data(user_data: dict):
    """Helper function to run all data prep for both models."""
    
    # --- Prep for Model A ---
    df_a_raw = pd.DataFrame([user_data])
    # Use the specific Model A categorical list
    df_a_encoded = pd.get_dummies(df_a_raw, columns=MODEL_A_CAT_FEATURES, dummy_na=False, drop_first=True)
    
    # Reindex to ensure all columns from training are present
    df_a = df_a_encoded.reindex(columns=repayment_features, fill_value=0)
    df_a.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_a.columns]

    # --- Prep for Model B (FIXED) ---
    df_b_raw = pd.DataFrame([user_data])
    
    # 1. One-Hot Encode the specific Model B categorical features
    df_b_encoded = pd.get_dummies(df_b_raw, columns=MODEL_B_CAT_FEATURES, dummy_na=False, drop_first=True)
    
    # 2. Reindex to match the training columns EXACTLY and fill missing *dummies* with 0
    df_b = df_b_encoded.reindex(columns=income_features, fill_value=0)
    
    # 3. Safety Fill for Continuous Features (Use 0 as a safe default for numeric inputs)
    df_b = df_b.fillna(0) 
    
    return df_a, df_b

def calculate_composite_score(user_data: dict):
    if not all([repayment_model, income_model]): return {"error": "ML models are not loaded."}
    
    try:
        df_a, df_b = get_prepared_data(user_data)
        
        # Model A Prediction
        # Predict_proba returns [Prob of No Default, Prob of Default]. We want Prob of No Default (Score)
        repayment_score = repayment_model.predict_proba(df_a)[:, 0][0] 
        
        # Model B Prediction (with log transform)
        log_prediction = income_model.predict(df_b)[0]
        predicted_value = np.expm1(log_prediction) # This is the predicted MPCE in Rupees
        
        # Normalize income score using a sigmoid function
        # Center point is the mean MPCE (~3588 from your data analysis)
        center_point = 3500  
        steepness = 1000     
        income_score = 1 / (1 + np.exp(-(predicted_value - center_point) / steepness))

    except Exception as e:
        return {"error": f"Model prediction failed. Details: {e}"}

    w1 = 0.6; w2 = 0.4
    composite_score = (w1 * repayment_score) + (w2 * income_score)
    
    # Risk Banding Logic
    # Low Risk (Good Repayment) = repayment_score > 0.7
    # High Need (Low Income) = income_score <= 0.5 (below the center point)
    LOW_RISK_THRESHOLD = 0.65
    if repayment_score > LOW_RISK_THRESHOLD and income_score <= 0.5: 
        risk_band = "Low Risk - High Need"  # The ideal candidate!
    elif repayment_score > LOW_RISK_THRESHOLD and income_score > 0.5: 
        risk_band = "Low Risk - Low Need"
    elif repayment_score <= LOW_RISK_THRESHOLD and income_score <= 0.5: 
        risk_band = "High Risk - High Need"
    else: 
        risk_band = "High Risk - Low Need"
    return {"repayment_score": round(float(repayment_score), 4), 
            "income_proxy_score": round(float(income_score), 4),
            "predicted_mpce": round(float(predicted_value), 2),
            "composite_score": round(float(composite_score), 4), 
            "risk_band": risk_band}

def get_shap_explanations(user_data: dict):
    # ... (Keep existing get_shap_explanations block - it is correct) ...
    if not all([repayment_explainer, income_explainer]): return {"error": "SHAP explainers are not loaded."}
    
    df_a, df_b = get_prepared_data(user_data)
    
    # SHAP for Model A
    shap_values_a = repayment_explainer.shap_values(df_a)
    
    # SHAP for Model B
    shap_values_b = income_explainer.shap_values(df_b)
    
    return {
        "repayment_explanation": {
            "base_value": repayment_explainer.expected_value[1], # Class 1 (Default)
            "shap_values": shap_values_a[1].tolist(),
            "feature_names": repayment_features,
            "feature_values": df_a.iloc[0].tolist()
        },
        "income_explanation": {
            "base_value": income_explainer.expected_value,
            "shap_values": shap_values_b[0].tolist(),
            "feature_names": income_features,
            "feature_values": df_b.iloc[0].tolist()
        }
    }