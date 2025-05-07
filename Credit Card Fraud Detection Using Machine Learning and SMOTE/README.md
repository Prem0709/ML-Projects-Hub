# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 Project Overview

This project is focused on detecting fraudulent credit card transactions using advanced machine learning techniques. Due to the extreme imbalance in the dataset (fraudulent transactions make up only ~0.17%), special emphasis was placed on **data preprocessing**, **class balancing (SMOTE)**, and **model evaluation**.

Implemented and evaluated both **Logistic Regression** with cross-validation and **Random Forest Classifier** using robust data processing and performance metrics like **ROC-AUC**, **Precision**, and **Accuracy**.

---

## 📁 Dataset Description

The dataset used is the **Credit Card Fraud Detection Dataset** from Kaggle.

- **Total records:** 284,807
- **Features:** 31 total (Time, Amount, V1-V28 (PCA-transformed), Class)
- **Target:**
  - `Class = 0`: Genuine transaction
  - `Class = 1`: Fraudulent transaction

---

## 🛠️ Workflow

### 1. **Data Loading & Initial Inspection**
- Checked for null values, data types, and duplicate entries.
- Removed any duplicate rows to maintain data integrity.

### 2. **Class Imbalance Analysis**
- Found only ~0.17% fraudulent transactions.
- Visualized distribution using bar plots.

### 3. **Feature Engineering & Preprocessing**
- Plotted histograms to understand feature distributions.
- Handled skewness in features using **Yeo-Johnson Transformation** (`PowerTransformer`).
- Normalized `Amount` feature using **RobustScaler**.

### 4. **Class Imbalance Handling**
- Used **SMOTE (Synthetic Minority Over-sampling Technique)** to oversample the minority class.
- Set `sampling_strategy=0.1` to balance classes to a 9:1 ratio.

### 5. **Model Training and Evaluation**

#### 🔷 Logistic Regression (with Stratified K-Fold)
- Tested different regularization strengths `C = [0.01, 0.1, 0.5, 1, 2]`.
- Evaluated using:
  - **Accuracy**
  - **Precision**
  - **ROC-AUC**

#### 🔷 Random Forest Classifier
- Parameters: `n_estimators=30`, `max_depth=30`, `max_samples=0.2`, `bootstrap=True`
- Evaluation Metrics:
  - Accuracy: 0.99+
  - ROC-AUC: High discriminative power
- Used **Partial Dependence Plots** to interpret feature importance.

---

## 📈 Results

| Model                 | Accuracy | Precision | ROC-AUC |
|----------------------|----------|-----------|---------|
| Logistic Regression  | ~98%     | Varies    | High    |
| Random Forest        | ~99%     | High      | High    |

- Random Forest performed better in terms of both **Accuracy** and **ROC-AUC**.
- SMOTE significantly improved the model’s ability to detect minority class (fraud).

---

## 📊 Visualizations

- **Class Distribution**: Before and after SMOTE
- **Feature Histograms**: To explore data distributions
- **Scatter Plots**: `Time vs Class`, `Amount vs Class`
- **Partial Dependence Plots**: Interpreting Random Forest on `Time`, `V1`, and `V2`

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Prem0709/ML-Projects-Hub.git
   cd ML-Projects-Hub/Credit-Card-Fraud-Detection
