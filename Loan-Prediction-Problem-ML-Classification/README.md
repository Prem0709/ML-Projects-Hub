# 🏦 Loan Prediction Problem

This project focuses on building a machine learning model that predicts whether a loan should be approved or not, based on various applicant attributes. It uses classification algorithms to provide predictive insights for banking and financial decision-making.

## 📁 Project Structure

- **`Loan_Prediction_Problem.ipynb`** – Complete Jupyter notebook with EDA, preprocessing, model training, and evaluation.
- **`train.csv`** – Training dataset with labeled outcomes.
- **`test.csv`** – Test dataset for final prediction.

---

## 📌 Problem Statement

Banks receive numerous loan applications every day. The goal is to automate the loan approval process using historical data. The model predicts loan approval (`Loan_Status`) based on attributes such as gender, marital status, income, education, credit history, etc.

---

## 🛠️ Technologies Used

- **Python**  
- **Pandas, NumPy** – Data manipulation  
- **Matplotlib, Seaborn** – Data visualization  
- **Scikit-learn** – Machine learning models  
- **Jupyter Notebook**

---

## 🔍 Exploratory Data Analysis (EDA)

The dataset was examined for:
- Null values and data imbalances
- Distribution of categorical features
- Correlation between features
- Outliers and skewness

---

## 🔄 Data Preprocessing

Steps included:
- Handling missing values
- Encoding categorical features
- Feature scaling
- Splitting data into training and testing sets

---

## 🤖 Models Used

Several models were trained and evaluated:
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Decision Tree**
- **Random Forest**
- **XGBoost**
- **Extra Trees Classifier**

Each model was assessed using:
- Accuracy
- Cross-validation
- Confusion Matrix
- Bias and Variance analysis

---

## ✅ Best Model

📌 **Logistic Regression (Tuned)**
- Test Accuracy: **83.74%**
- Cross-Validation Accuracy: **79.8%**
- Penalty: `l1`
- Solver: `liblinear`

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/loan-prediction-ml.git
   cd loan-prediction-ml
   ```
2. Open the notebook:
   ```bash
   jupyter notebook Loan_Prediction_Problem.ipynb
   ```
3. Run all cells and observe the outputs.

---

## 📈 Output

The model predicts whether a loan will be **Approved (Y)** or **Rejected (N)** for new applicants based on input features.

---

## 📬 Contact

For any queries or collaboration, feel free to reach out:  
📧 [Your Email]  
🔗 [LinkedIn Profile or Portfolio]

---
