# main.py (Final Corrected Version for Deployment)

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from scorer import calculate_composite_score, get_shap_explanations

app = FastAPI(
    title="SIH Beneficiary Credit Scoring API",
    description="An API that uses two ML models to predict a composite credit score and provide explanations.",
    version="FINAL"
)

# --- Define the EXACT INPUTS the API expects from the user ---
class ApplicantData(BaseModel):
    # --- Features for Model A (Repayment Model - ALL features from your training script) ---
    Age: int = Field(..., example=45)
    Income: int = Field(..., example=500000)
    LoanAmount: int = Field(..., example=100000)
    CreditScore: int = Field(..., example=750)
    MonthsEmployed: int = Field(..., example=120)
    NumCreditLines: int = Field(..., example=5)
    InterestRate: float = Field(..., example=12.5)
    LoanTerm: int = Field(..., example=60)
    DTIRatio: float = Field(..., example=0.3)
    
    # Model A Categorical Features
    Education: str = Field(..., example="Bachelor's")
    EmploymentType: str = Field(..., example="Full-time")
    MaritalStatus: str = Field(..., example="Married")
    HasMortgage: str = Field(..., example="Yes")
    HasDependents: str = Field(..., example="Yes")
    LoanPurpose: str = Field(..., example="Debt Consolidation")
    HasCoSigner: str = Field(..., example="No")
    
    # --- Features for Model B (Income Proxy Model - R2=0.65, Leak-free features) ---
    
    # Demographics/Socio-Economic (Categorical)
    Sector: str = Field(..., example="Rural", description="Rural or Urban")
    Social_Group_of_HH_Head: str = Field(..., example="OBC", description="Social Group (e.g., SC/ST/OBC/Other)")
    Max_Income_Activity: str = Field(..., example="Self-Employed", description="Primary occupation type")
    Type_of_Dwelling_Unit: str = Field(..., example="Owned", description="Dwelling status (e.g., Owned, Hired)")
    Land_Ownership: str = Field(..., example="Yes", description="Owns agricultural land (Yes/No)")
    Ration_Card_Type: str = Field(..., example="Priority", description="Type of Ration Card")
    Religion_of_HH_Head: str = Field(..., example="Hindu", description="Religion of Household Head")

    # Continuous/Engineered (Numeric)
    head_of_household_age: int = Field(..., example=40)
    household_size_calculated: int = Field(..., example=5)
    avg_education_years_adults: float = Field(..., example=8.5)
    num_internet_users: int = Field(..., example=1)
    
    # Consumption Proxies
    fuel_expenditure: float = Field(..., example=1250.0, description="Avg monthly cost of electricity/fuel")
    comm_expenditure: float = Field(..., example=450.0, description="Avg monthly cost of mobile recharge/bills")
    
    # Engineered Score Proxies (Direct input for composite features)
    Asset_Score_X1: float = Field(..., example=7.0, description="Composite score based on possessions (0-24)")
    Scheme_Index_X2: float = Field(..., example=2.15, description="Composite score for reliance on social schemes (0-5)")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Credit Scoring API!"}

@app.post("/score")
def get_score(data: ApplicantData):
    return calculate_composite_score(data.dict())

@app.post("/explain")
def get_explanation(data: ApplicantData):
    return get_shap_explanations(data.dict())