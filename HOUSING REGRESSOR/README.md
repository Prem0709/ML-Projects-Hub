You're working on a **House Price Prediction Web App** built using Flask and multiple machine learning models. Here's a breakdown of what your project is doing and how to present it on GitHub.

---

### 🔍 **Project Summary for GitHub**
You can use the following information in your `README.md` file:

---

## 🏡 House Price Prediction Web App

This project is a web-based application built using **Flask** that allows users to predict house prices based on several features like income, house age, number of rooms, etc. It compares multiple **machine learning regression models** for performance and accuracy.

### 🚀 Features

- Web interface using **Flask** and **HTML templates**
- Predict house prices using different regression models
- Supports 13 different ML models
- Displays performance metrics of each model
- Input form for features like income, age, number of rooms, etc.

### 🧠 Models Used

The app includes the following regression models:

- Linear Regression ✅
- Robust Regression ✅
- Ridge Regression ✅
- Lasso Regression ✅
- ElasticNet ✅
- Polynomial Regression ✅
- SGD Regressor ✅
- Artificial Neural Network (ANN) ✅
- Random Forest ✅
- Support Vector Machine (SVM) ✅
- LightGBM (LGBM) ✅
- XGBoost ✅
- K-Nearest Neighbors (KNN) ✅

All models are pre-trained and loaded from `.pkl` files for fast prediction.

### 📁 Project Structure

```
├── app.py                  # Main Flask application
├── templates/
│   ├── index.html          # Homepage with input form
│   └── results.html        # Prediction results page
│   └── model.html          # Model performance comparison
├── static/                 # Optional: CSS, JS files
├── pkl/                    # Pre-trained models (Pickle files)
├── model_evaluation_results.csv # Model performance metrics
```

### 🛠 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-predictor.git
   cd house-price-predictor
   ```

2. Set up a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate for Windows
   ```

3. Install dependencies:
   ```bash
   pip install flask pandas scikit-learn xgboost lightgbm
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Visit `http://localhost:5000` in your browser.

---
