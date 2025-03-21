# Salary Prediction Using Linear Regression

## Introduction
This project demonstrates the implementation of a **Linear Regression** model to predict salaries based on years of experience. The dataset is analyzed with various statistical methods, and key evaluation metrics are calculated to assess model performance.

## Dataset Description
The dataset contains the following attributes:
- **YearsExperience**: Number of years a person has worked.
- **Salary**: Corresponding salary in currency units.

## Key Steps in the Project
1. **Data Preprocessing**
   - Load the dataset
   - Check for missing values and outliers
   - Perform exploratory data analysis (EDA)
   
2. **Feature Engineering**
   - Compute summary statistics (mean, median, mode, variance, standard deviation)
   - Analyze correlation and skewness
   
3. **Model Training**
   - Split the dataset into training and testing sets
   - Train a **Linear Regression** model
   - Obtain coefficients and intercept
   
4. **Model Evaluation**
   - Calculate bias-variance tradeoff
   - Evaluate using **R-Squared, Mean Squared Error (MSE), Mean Absolute Error (MAE)**
   - Predict salary for new experience levels

## Important Formulas
Below are key formulas used in regression analysis:

1. **Linear Regression Equation:**
   \[ y = mx + c \]
   - **y**: Predicted salary
   - **m**: Coefficient (slope)
   - **x**: Years of experience
   - **c**: Intercept

2. **Mean (Average):**
   \[ \mu = \frac{\sum x_i}{n} \]
   - **n**: Total number of data points

3. **Variance (\( \sigma^2 \)):**
   \[ \sigma^2 = \frac{\sum (x_i - \mu)^2}{n} \]
   
4. **Standard Deviation (\( \sigma \))**:
   \[ \sigma = \sqrt{\sigma^2} \]
   
5. **Mean Squared Error (MSE):**
   \[ MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \]
   
6. **Root Mean Squared Error (RMSE):**
   \[ RMSE = \sqrt{MSE} \]
   
7. **Mean Absolute Error (MAE):**
   \[ MAE = \frac{1}{n} \sum |y_i - \hat{y}_i| \]
   
8. **R-Squared (Coefficient of Determination):**
   \[ R^2 = 1 - \frac{SSR}{SST} \]
   - **SSR (Sum of Squares of Residuals):** \( \sum (y_i - \hat{y}_i)^2 \)
   - **SST (Total Sum of Squares):** \( \sum (y_i - \bar{y})^2 \)

9. **Z-Score (Standardization):**
   \[ Z = \frac{x - \mu}{\sigma} \]
   - Measures how many standard deviations a data point is from the mean.

10. **Bias-Variance Tradeoff:**
   \[ Bias^2 + Variance + Irreducible Error = Total Error \]
   - **Bias**: Error due to overly simplistic model assumptions
   - **Variance**: Error due to model sensitivity to training data
   
## Model Performance
- **Regression Coefficient (Slope):** `9312.57`
- **Intercept:** `26780.10`
- **R-Squared on Training Data:** `0.9411`
- **R-Squared on Testing Data:** `0.9881`
- **Future Salary Prediction (20 years experience):** `213020`

## Learning Notes for Beginners
- Linear regression finds the best-fit line by minimizing the error between actual and predicted values.
- **Correlation coefficient (R)** determines the strength of the relationship between two variables.
- **Feature Scaling (Normalization or Standardization)** helps when working with different data scales.
- **Outliers** can significantly impact regression models, so they should be detected and handled properly.
- **Overfitting** occurs when a model performs well on training data but poorly on test data; using cross-validation can help.
- **Multicollinearity** can affect model stability; checking correlation matrices helps detect it.

## Future Improvements
- Implement **Polynomial Regression** to capture nonlinear trends.
- Apply **Ridge and Lasso Regression** to handle overfitting.
- Use **Multiple Linear Regression** to add more relevant features.
- Explore **Deep Learning models** for improved salary predictions.

## Conclusion
This project provides a solid foundation for beginners to understand regression analysis, interpret results, and apply statistical techniques effectively. Happy Learning! 🚀

