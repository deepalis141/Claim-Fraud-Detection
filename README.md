# Claim-Fraud-Detection
Machine learning project to detect fraudulent insurance claims using classification models, SMOTE imbalance handling, Isolation Forest anomaly detection, business rules, and hybrid fraud scoring. Includes Tableau dashboard and full end-to-end pipeline.




# 🕵️ Claim Fraud Detection

### **Machine Learning + Isolation Forest + Rules Engine + Tableau Dashboard**

---

## 📌 **Objective**

Detect **fraudulent or suspicious insurance claims** using:

* Classification models (Logistic Regression, Random Forest)
* Imbalanced learning techniques (class weights, SMOTE)
* Anomaly detection (Isolation Forest)
* Business rule scoring
* Hybrid fraud scoring
* Tableau dashboard for risk visualization

This project simulates a real insurance fraud analytics workflow.

---

## 📂 **Project Structure**

```
fraud_detection_project/
│
├── data/
│   └── creditcard.csv                 # dataset downloaded from Kaggle
│
├── notebooks/
│   ├── 01_fraud_eda_cleaning.ipynb    # EDA + preprocessing
│   ├── 02_fraud_modeling.ipynb        # ML models + Isolation Forest + scoring
│   └── 03_fraud_tableau_export.ipynb  # output CSV for Tableau dashboard
│
├── models/
│   ├── fraud_rf_model.pkl             # trained RandomForest model
│   └── fraud_scaler.pkl               # StandardScaler for preprocessing
│
├── dashboard/
│   └── fraud_dashboard_data.csv       # final dataset for Tableau
│
└── README.md
```

---

## 📊 **Dataset**

For this project, the **Credit Card Fraud Detection** dataset is used, as it closely mimics fraud patterns:

* Highly imbalanced (fraud ≈ 0.17%)
* Transaction features (V1–V28 from PCA)
* Amount, Time, Class (0 = normal, 1 = fraud)

📥 Kaggle link:
[https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

> In a real insurance dataset, features would include policy age, claim history, severity, repair garage flags, etc.
> The modeling approach here is identical.

---

## 🧹 **1. Data Cleaning & Fraud EDA (Notebook 1)**

Performed in `01_fraud_eda_cleaning.ipynb`:

### ✔ Loaded and inspected dataset

* Checked missing values
* Verified fraud imbalance
* Visualized:

  * **class distribution**
  * **transaction amount vs fraud**
  * **time-of-day fraud patterns**

### ✔ Saved processed dataset

Output:
`data/fraud_processed.csv`

---

## 🧠 **2. Feature Engineering**

Applied inside Notebook 2:

✔ Scaled numerical features with **StandardScaler**
✔ Created domain-inspired "rule features" such as:

* high amount indicator
* early-day transaction indicator

> In real insurance datasets you would add:
>
> * policy age
> * number of past claims
> * severity ratio
> * garage/hospital flags
> * accident type indicators

---

## 🤖 **3. Machine Learning Models (Classification)**

Trained 3 models:

### 🔹 Logistic Regression

* Used `class_weight="balanced"`
* Handles imbalance
* Good baseline model

### 🔹 Random Forest

* `class_weight="balanced"`
* More robust than Logistic Regression
* Higher recall on fraud class

### 🔹 Random Forest + SMOTE

* Oversampled fraud cases
* Best balance between precision/recall
* **Selected as primary ML model**

### 🔹 Evaluation Metrics

* Precision
* Recall
* F1-score
* ROC–AUC

---

## 🧩 **4. Imbalanced Data Handling**

Fraud datasets are extremely skewed.

We applied:

### ✔ Class weights

(LR & RF trained with `class_weight="balanced"`)

### ✔ SMOTE Oversampling

* Generates synthetic fraud examples
* Improved fraud recall significantly

---

## 🚨 **5. Isolation Forest (Anomaly Detection)**

An unsupervised model trained to detect unusual transactions.

* Flags anomalies with `-1`
* Combines well with supervised ML
* Gives anomaly score (`iso_anomaly_score`)
* Helps catch unseen fraud patterns

---

## 🧮 **6. Business Rules Engine**

A simple rule-based system:

### Example rules used:

* High transaction amount
* Transaction in early window (first hour)

Rules generate a **rule_score** between 0 and 1.

> In actual insurance datasets, rules include:
> duplicate claims, mismatched documents, abnormal repair costs, clinic red flags, etc.

---

## 🔢 **7. Hybrid Scoring System**

Combines ML predictions + rule-based behavior:

### Final score:

```
fraud_score = 0.6 * ML_prob + 0.4 * rule_score
```

This creates a powerful numeric ranking of suspicious claims.

---

## 🏅 **8. Ranking Top Suspicious Claims**

Created a dataframe:

| fraud_score | ml_prob | rule_score | iso_anomaly_score | actual_class |

Then sorted descending:

```python
fraud_results_sorted = fraud_results.sort_values("fraud_score", ascending=False)
```

Exported:

* **Top 50 risky transactions**
* **Full fraud dashboard CSV**

---

## 📈 **9. Tableau Dashboard**

Built using `dashboard/fraud_dashboard_data.csv`.

Contains:

### 🔹 Heatmap

`fraud_score` vs `Amount` / `Time`

### 🔹 Suspicious Transactions Table

Ranked by fraud score

### 🔹 Fraud vs Non-Fraud Segments

Compare distributions
Identify high-risk patterns

### 🔹 Anomaly Map

IsolationForest’s outliers highlighted

This visually explains fraud behavior and helps business teams act fast.

---

## ⚙️ **Tech Stack**

* Python (pandas, numpy)
* scikit-learn
* imbalanced-learn (SMOTE)
* Isolation Forest
* Tableau (Dashboarding)
* Jupyter Notebook
* GitHub

---

## 🚀 **How to Run (Ubuntu + Miniconda)**

### 1️⃣ Create Conda environment

```bash
conda create -n fraud_env python=3.10 -y
conda activate fraud_env
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Place dataset

Download from Kaggle and place in:

```
data/creditcard.csv
```

### 4️⃣ Run notebooks in order

1. `01_fraud_eda_cleaning.ipynb`
2. `02_fraud_modeling.ipynb`
3. `03_fraud_tableau_export.ipynb`

---

## 📝 **Results**

* Built a complete fraud detection pipeline
* Handled severe class imbalance
* Combined supervised + unsupervised models
* Created hybrid fraud scoring
* Produced business-friendly Tableau dashboard
* Fully documented in GitHub-friendly structure

---

## 🙌 **Author**

Deepali Sharma
https://www.linkedin.com/in/deepali007



Just tell me!
