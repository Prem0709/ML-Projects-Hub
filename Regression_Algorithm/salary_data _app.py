import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

def main():
    st.title("Salary Prediction Based on Experience")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview:")
        st.write(dataset.head())
        
        x = dataset.iloc[:, :-1].values.reshape(-1, 1)
        y = dataset.iloc[:, -1].values
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        
        coef = regressor.coef_[0]
        intercept = regressor.intercept_
        
        st.write(f"### Model Coefficients:")
        st.write(f"Coefficient: {coef}")
        st.write(f"Intercept: {intercept}")
        
        # Visualization
        fig, ax = plt.subplots()
        ax.scatter(x_test, y_test, color='red', label='Actual')
        ax.plot(x_train, regressor.predict(x_train), color='blue', label='Predicted')
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_title("Salary vs Experience (Test Set)")
        ax.legend()
        st.pyplot(fig)
        
        # Prediction based on user input
        exp = st.number_input("Enter Years of Experience for Prediction:", min_value=0.0, step=0.1)
        if exp:
            pred_salary = regressor.predict(np.array([[exp]]))[0]
            st.write(f"### Predicted Salary: {pred_salary:.2f}")
        
        # Model Accuracy
        train_score = regressor.score(x_train, y_train)
        test_score = regressor.score(x_test, y_test)
        
        st.write(f"### Model Performance:")
        st.write(f"Training Score (R^2): {train_score:.4f}")
        st.write(f"Test Score (R^2): {test_score:.4f}")
        
        # Statistical Analysis
        st.write("### Dataset Statistics:")
        st.write(dataset.describe())
        st.write("Skewness:", dataset.skew())
        st.write("Correlation Matrix:")
        st.write(dataset.corr())
        
if __name__ == "__main__":
    main()
