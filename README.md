# 🚀 SIH Credit Scoring System

**A Dual-Model Credit Risk & Income Assessment System with FastAPI +
Streamlit + SHAP**

This repository contains the complete implementation of an end-to-end
credit scoring pipeline built for **Smart India Hackathon (SIH)**. It
integrates **two ML models**, a **composite scoring engine**, a
**FastAPI backend**, and a **Streamlit dashboard**.

------------------------------------------------------------------------

# 🧩 1. Project Overview

The system evaluates an applicant's **creditworthiness** using a
dual-model architecture:

## 🔹 **Model A --- Repayment Prediction Model (XGBoost Classifier)**

Predicts the probability that a borrower will **NOT default** on their
loan.

Input features include: - Age, Employment, Loan Amount, Credit Score\
- Loan Purpose\
- Categorical borrower attributes (One-hot encoded)

**Output:**\
- `repayment_score = P(No Default)`

------------------------------------------------------------------------

## 🔹 **Model B --- Income Proxy Model (XGBoost Regressor)**

Predicts **Monthly Per Capita Expenditure (MPCE)** using socio-economic
indicators derived from the **HCES 2023--24 dataset**.

Uses engineered household features: - Household size, education,
dwelling type
- Fuel & communication expenditure
- Asset_Score_X1
- Scheme_Index_X2
- Religion, Social Group, Occupation
- Land Ownership, Ration Card Type

**Output:**
- `predicted_mpce (₹)`
- `income_proxy_score` (normalized sigmoid score)

------------------------------------------------------------------------

## 🔹 **Composite Scoring System**

    Composite Score = (0.6 × Repayment Score) + (0.4 × Income Proxy Score)

### 🎯 Risk Banding

  Repayment   Income   Risk Band
  ----------- -------- ------------------------
  High        Low      Low Risk -- High Need
  High        High     Low Risk -- Low Need
  Low         Low      High Risk -- High Need
  Low         High     High Risk -- Low Need

------------------------------------------------------------------------

# 🏗️ 2. Repository Structure

    ├── data_assembly/
    │   └── HCES_Data_Processing.ipynb
    │
    ├── Model_A_data/
    │   └── Loan_default.csv
    │
    ├── saved_models/
    │   ├── repayment_model_xgb.joblib
    │   ├── repayment_model_features.joblib
    │   ├── repayment_model_explainer.joblib
    │   ├── income_model_final.joblib
    │   ├── income_model_final_features.joblib
    │   └── income_model_final_explainer.joblib
    │
    ├── streamlit_app/
    │   └── app.py
    │
    ├── main.py
    ├── scorer.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# 🔧 3. Setup Instructions

## **1. Clone the repository**

``` bash
git clone <https://github.com/yashkumar181/Samriddhi-Credit_Scoring_Automation/tree/main>
cd <Samriddhi-Credit_Scoring_Automation>
```

## **2. Install Dependencies**

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🚀 4. Running FastAPI

``` bash
uvicorn main:app --reload
```
------------------------------------------------------------------------

# 📤 5. Example Request Body

``` json
{
  "Age": 45,
  "Income": 500000,
  "LoanAmount": 100000,
  "CreditScore": 750,
  "MonthsEmployed": 120,
  "NumCreditLines": 5,
  "InterestRate": 12.5,
  "LoanTerm": 60,
  "DTIRatio": 0.3,
  "Education": "Bachelor's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": "Yes",
  "HasDependents": "Yes",
  "LoanPurpose": "Debt Consolidation",
  "HasCoSigner": "No",
  "Sector": "Rural",
  "Social_Group_of_HH_Head": "OBC",
  "Max_Income_Activity": "Self-Employed",
  "Type_of_Dwelling_Unit": "Owned",
  "Land_Ownership": "Yes",
  "Ration_Card_Type": "Priority",
  "Religion_of_HH_Head": "Hindu",
  "head_of_household_age": 40,
  "household_size_calculated": 5,
  "avg_education_years_adults": 8.5,
  "num_internet_users": 1,
  "fuel_expenditure": 1250,
  "comm_expenditure": 450,
  "Asset_Score_X1": 7,
  "Scheme_Index_X2": 2.15
}
```

# 🧠 6. ML Models

### 🟦 Model A: Repayment Model (XGBoost Classifier)
- Predicts probability of **successful loan repayment** (P(No Default))
- Trained on Loan Default dataset (`Loan_default.csv`)
- Uses demographic, credit, and financial inputs
- Fully supports **SHAP explainability**
- Outputs:  
  - `repayment_score`

---

### 🟩 Model B: Income Proxy Model (XGBoost Regressor)
- Predicts **MPCE (Monthly Per Capita Expenditure)** as a proxy for household income
- Trained on processed HCES dataset (`master_dataset.parquet`)
- Uses engineered socio-economic & consumption features
- Applies log-transform during training → outputs converted using `expm1`
- Leak-free feature engineering (no MPCE-derived inputs)
- Outputs:  
  - `predicted_mpce`  
  - `income_proxy_score` (normalized using sigmoid)

---
------------------------------------------------------------------------

# 📊 7. Streamlit Dashboard

``` bash
streamlit run streamlit_app/app.py
```

------------------------------------------------------------------------

# 🖼️ 8. Dashboard Screenshots

Add your screenshots here:

    ![Dashboard Screenshot 1](images/dashboard_1.png)
    ![Dashboard Screenshot 2](images/dashboard_2.png)

------------------------------------------------------------------------

# 🧪 9. HCES Data Assembly Pipeline (Final README Section)

This section describes the complete workflow used to generate the **master_dataset.parquet** file from the raw **HCES 2023–24 SPSS (.sav)** files.  
This dataset is the foundation for **Model B – Income Proxy Model**.

---

# 📦 Overview

The HCES pipeline performs:

- Merging HCES Levels **01–15**
- Cleaning household and person-level records
- Chunk-based item-level expenditure extraction
- Feature engineering (assets + scheme indexes)
- MPCE (Monthly Per Capita Expenditure) target construction
- Exporting a final modeling file:  
  ✔ `master_dataset.parquet`

---

# 🔄 Processing Workflow

## **1️⃣ Merging Household-Level SPSS Files**
All `.sav` files from Levels 01, 03, 04, 07, 11 are merged using keys:

- `FSU_Serial_No`
- `Panel`
- `Sub_sample`
- `Sample_Household_No`

Duplicates are removed, numeric types corrected, and memory usage reduced.

---

## **2️⃣ Level-15 Processing (Consumption Expenditure)**
`LEVEL - 15` contains:

- `MONTHLY_CONSUMPTION_EXP`
- `HOUSEHOLD_SIZE`

Steps:
1. Convert merge keys to clean formats  
2. Group by household  
3. Take max values to avoid NaN inconsistencies  
4. Merge into base dataset

---

## **3️⃣ Person-Level Feature Engineering (LEVEL 02)**
Aggregates created:

- **`household_size_calculated`** — count of persons  
- **`head_of_household_age`** — extracted via relation tag  
- **`avg_education_years_adults`** — mean for adults (age ≥18)  
- **`num_internet_users`** — count of 30-day active users

This creates household socio-economic richness.

---

## **4️⃣ Item-Level Expenditure Extraction (LEVEL 05–12)**
These files can contain millions of rows — so chunk processing is used:

### Workflow:
- Read `.sav` files in **1,000,000 row chunks**
- Identify:
  - `item_code` column  
  - `value` column  
- Write cleaned chunks to temp CSV  
- Aggregate items of interest:

### Fuel item codes: 
[332, 338, 331, 334, 335, 341, 343, 337, 333, 344, 345, 340, 336, 342]


### Communication item codes:
[488, 487, 496, 490]


Outputs:
- **`fuel_expenditure`**  
- **`comm_expenditure`**

---

# 🛠️ Engineered Features

## ⭐ `Asset_Score_X1`
Weighted possessions (example):

| Asset | Weight |
|-------|--------|
| Car, Truck | 5 |
| Washing Machine, Laptop | 3 |
| Refrigerator, TV | 2 |
| Scooter, Bicycle, Air Cooler | 1 |

Final score = weighted sum of all binary “Possess_” fields.

---

## ⭐ `Scheme_Index_X2`
Binary benefit indicators:

- PMGKY beneficiary  
- Ayushman  
- LPG subsidy  
- Medical assistance  
- Ration usage  

Index = sum of selected scheme columns.  
Indicates **government dependency**.

---

# 🎯 MPCE Target Variable
MPCE = MONTHLY_CONSUMPTION_EXP / household_size


Rows with missing consumption or division errors are removed.

---

# 📁 Final Output

After all cleaning + merges:

✔ **227,114 households**  
✔ 40+ clean features  
✔ Fully processed socio-economic dataset  
✔ Saved as: master_dataset.parquet


Used as input to:
- **Income Proxy Model (Model B)**
- SHAP explainability layer
- Composite Credit Scoring

---

# 🧷 Notes
- All SPSS reading uses `pyreadstat` for stability  
- Chunk-based processing prevents RAM overflow  
- All keys are normalized to consistent string format  
- Final dataset has **no duplicates** and **no missing MPCE**

---

------------------------------------------------------------------------


# 🤝 10. Contributors

- [Yash Kumar](https://github.com/yashkumar181)
    
- [Simon Mittal](https://github.com/Simon-Mittal)
    
- [Shweta Kumari](https://github.com/shwetakkk21)
    
- [Tarun Sharma](https://github.com/TARUN-CO) 

------------------------------------------------------------------------

# ⚖️ License

MIT License
