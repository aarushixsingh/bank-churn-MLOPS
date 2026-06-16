# bank-churn-MLOPS


cors cross origin resource sharing
# 🏦 Bank Customer Churn Prediction — End-to-End ML Project

> A production-ready machine learning system that predicts whether a bank customer will churn, explains *why* using SHAP, and serves predictions via a REST API with a live interactive frontend.

🔗 **[Live Demo](https://bank-churn-mlops-eta.vercel.app/)** | 📦 **[API Docs](https://bank-churn-mlops.onrender.com/docs)**

---

## 🚀 What This Project Does

A bank employee enters a customer's details into the web app. The system instantly:
1. Predicts whether the customer will churn or stay
2. Shows the probability and risk level (Low / Medium / High)
3. Explains **in plain English** which factors drove the prediction
4. Displays a color-coded feature impact chart (powered by SHAP)

---

## 🖼️ Demo

| Input Form | Prediction Result |
|---|---|
| Fill in customer details | Instant verdict with explanation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Models** | logistic-regression, XGBoost, Random Forest |
| **Explainability** | SHAP (SHapley Additive Explanations) |
| **Hyperparameter Tuning** | GridSearchCV |
| **API Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML, CSS, JavaScript |
| **Model Persistence** | Joblib |
| **Frontend Deployment** | Vercel |
| **Backend Deployment** | Render |

---

## 📊 Dataset

- **Source:** [Bank Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)
- **Size:** 10,000 customers, 12 features
- **Target:** `churn` (1 = churned, 0 = stayed)
- **Class imbalance:** 79.6% stayed vs 20.4% churned

---

## 🧠 ML Pipeline

### 1. Exploratory Data Analysis
- Analyzed distributions of age, balance, credit score, and tenure against churn
- Key finding: customers aged 40–50 and inactive members showed significantly higher churn rates
- Customers with high balances are more likely to churn — they have more incentive to switch banks

### 2. Ethical Feature Selection
- Deliberately **excluded `gender` and `country`** from the model to prevent algorithmic bias
- Using gender or geography in financial predictions can lead to discriminatory outcomes and is illegal in many jurisdictions
- Model decisions are based purely on behavioural and financial signals

### 3. Class Imbalance Handling
- Rejected SMOTE (synthetic data introduces noise and doesn't guarantee correctness)
- Used **class weights** instead — penalizes the model more for missing actual churners
- XGBoost: `scale_pos_weight = 7963/2037 ≈ 3.9`
- Sklearn models: `class_weight='balanced'`

### 4. Models Trained & Compared

| Model | Recall (class 1) | F1 (class 1) | Precision (class 1) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.36 | 0.48 | 0.70 |
| Random Forest | 0.74 | 0.50 | 0.38 |
| XGBoost | 0.76 | 0.57 | 0.46 |
| **Tuned Random Forest** ✅ | **0.72** | **0.58** | **0.48** |
| Tuned XGBoost | 0.76 | 0.57 | 0.46 |

### 5. Why Random Forest Was Chosen
- Best **F1 score (0.58)** — most balanced between precision and recall
- Higher precision than XGBoost — fewer false alarms (unnecessary retention offers)
- More interpretable than XGBoost for business stakeholders

### 6. Evaluation Metric Choice
- Primary metric: **Recall** for hyperparameter tuning (catching churners matters more than false alarms)
- Reporting metric: **F1 score** (standard for imbalanced classification)
- Avoided accuracy — misleading on imbalanced datasets (a model predicting "stay" always gets 80%)

### 7. Hyperparameter Tuning
Used `GridSearchCV` with 5-fold cross validation:

**Random Forest best params:**
```
n_estimators: 200
max_depth: 5
min_samples_split: 2
class_weight: balanced
```

---

## 🔍 Explainability with SHAP

Every prediction comes with a SHAP explanation showing which features pushed the model toward churn or staying:

- **Positive SHAP value** → feature increases churn probability
- **Negative SHAP value** → feature decreases churn probability

Example output for a high-risk customer:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.77,
  "feature_impact": {
    "age": 0.163,
    "active_member": 0.058,
    "products_number": 0.033,
    "balance": 0.019,
    "credit_score": -0.0001
  }
}
```

Age and inactivity were the biggest churn drivers — consistent with EDA findings.

---

## 📁 Project Structure

```
bank-churn-mlops/
├── data/
│   └── Churn_Modelling.csv
├── notebooks/
│   └── 01_eda.ipynb          # EDA, feature engineering, model training
├── api/
│   ├── main.py               # FastAPI backend with SHAP explainability
│   └── index.html            # Frontend UI
├── models/
│   └── random_forest_model.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/aarushixsingh/bank-churn-MLOPS.git
cd bank-churn-MLOPS
```

**2. Create virtual environment**
```bash
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1   # Windows
source myvenv/bin/activate       # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
pip install shap
```

**4. Run the API**
```bash
cd api
uvicorn main:app --reload
```

**5. Open the frontend**

Open `api/index.html` in your browser. Make sure the API is running on port 8000.

Or visit the API docs at: `http://localhost:8000/docs`

---

## 🌐 Deployment

| Service | Platform | URL |
|---|---|---|
| Frontend | Vercel | [bank-churn-mlops-eta.vercel.app](https://bank-churn-mlops-eta.vercel.app/) |
| Backend API | Render | your-render-url.onrender.com |

---

## 💡 Key Design Decisions

| Decision | Reasoning |
|---|---|
| Dropped gender & country | Prevents algorithmic bias in financial predictions |
| Class weights over SMOTE | Avoids synthetic data noise, more honest evaluation |
| Random Forest over XGBoost | Better F1, higher precision, more interpretable |
| SHAP over feature importance | Per-prediction explanations, not just global averages |
| Recall for tuning, F1 for reporting | Business context: missing a churner is costlier than a false alarm |

---

## 📈 Future Improvements

- [ ] Add MLflow experiment tracking
- [ ] Retrain pipeline with new data via scheduled jobs
- [ ] Add model drift monitoring (Evidently AI)
- [ ] Dockerize for consistent deployment
- [ ] Add batch prediction endpoint for bulk customer scoring

---

## 👨‍💻 Author

**Aarushi Singh**
- GitHub: [@aarushixsingh](https://github.com/aarushixsingh)
