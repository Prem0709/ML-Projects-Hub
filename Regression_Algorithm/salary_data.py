import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"C:\Users\pawar\python_data_science\ML\Salary_Data.csv")

x= dataset.iloc[:, :-1]
y= dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)



x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test) 

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')

plt.title("Salary Vs Experiance (Test set)")
plt.xlabel("Year of Experiance")
plt.ylabel("Salary ")
plt.show()

plt.scatter(x_train, y_train, color = 'blue')
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title("Salary Vs Experiance (Test set)")
plt.xlabel("Year of Experiance")
plt.ylabel("Salary ")
plt.show()


coef = print(f"Coefficientt : {regressor.coef_}")
intercept = print(f"Coefficientt : {regressor.intercept_}")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)


exp_12_future_pred = 9312 * 20 + 26780
exp_12_future_pred


bias = regressor.score(x_train, y_train)
print(bias)
variance = regressor.score(x_test, y_test)
print(variance)


dataset.mean()
dataset['Salary'].mean()
dataset['Salary'].median()
dataset['Salary'].mode()
dataset.var()
dataset['Salary'].var()
dataset['Salary'].std()
dataset.std()

from scipy.stats import variation
variation(dataset.values)
from scipy.stats import variation
variation(dataset['Salary'].values)

dataset.corr()
dataset['Salary'].corr(dataset['Salary'])

dataset.skew()
dataset['Salary'].skew()

dataset['Salary'].sem()
dataset.sem()

import scipy.stats as stats
dataset.apply(stats.zscore)

a = dataset.shape[0] 
b = dataset.shape[1] 
degree_of_freedom = a - b
degree_of_freedom = len(y_test) - 2  
print(f"Degrees of Freedom: {degree_of_freedom}")


#
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y= y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total=np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)


r_square = 1- (SSR/SST)
r_square+0
