## Import the necessary libraries  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

## Loading the Dataset 
dataset = pd.read_csv(r"C:\Users\pawar\data_science\ML\classification\Loan Prediction Problem\train_u6lujuX_CVtuZ9i.csv")
test_data = pd.read_csv(r"C:\Users\pawar\data_science\ML\classification\Loan Prediction Problem\test_Y3wMUE5_7gLdaTN.csv")

print(dataset.head(5))
print(test_data.head(5))

print(dataset.shape)
print(test_data.shape)

print(dataset.describe())
print(test_data.describe())

print(dataset.isna().sum())
print(test_data.isna().sum())

## Fill Missing Values In Training Set 
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)

## Fill Missing Values In Testing Set 
test_data['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
test_data['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
test_data['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)
test_data['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
test_data['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
test_data['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)

## Exploratory Data Analysis 
### Categorical Attributes visualization 
categorical = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Loan_Status']
for col in categorical:  
    sns.countplot(x=col, data=dataset, palette='Set2')
    plt.title(f"Chart for {col}") 
    plt.legend()
    plt.tight_layout()
    plt.show()

### Numercal Attributes visualization 
numerical = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History']
for col in numerical:  
    sns.histplot(x=col, data=dataset, kde=True)
    plt.title(f"Chart for {col}") 
    plt.legend()
    plt.tight_layout()
    plt.show()

### Corelation matrix 
corr = dataset.corr(numeric_only=True)
Fib,ax = plt.subplots(figsize=(12,8))
#### sns.heatmap(corr, annot=True, ax=ax, cmap='BuPu')
sns.heatmap(corr, annot=True)
plt.title('Heatmap of Correlation Matrix')
corr

## Drop Unneccesssary Column
train_dataset = dataset.drop(columns=['Loan_ID', 'Gender', 'Education', 'Married', 'Self_Employed'])
print(train_dataset.head())
test_data_2 = test_data.drop(columns=['Loan_ID', 'Gender', 'Education', 'Married', 'Self_Employed'])
print(test_data_2.head())

df_train = train_dataset.copy()
df_test =  test_data_2.copy()

## LabelEncoder
from sklearn.preprocessing import LabelEncoder
categorical_new = ['Dependents', 'Property_Area']
label_encoders = {}
for col in categorical_new:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])      # Fit + transform on train
    df_test[col] = le.transform(df_test[col])            # Only transform on test
    label_encoders[col] = le    
                             # Save the encoder
LabelEncoder_train = LabelEncoder()
df_train['Loan_Status'] = LabelEncoder_train.fit_transform(df_train['Loan_Status'])

### Train- Test Split 
X = df_train.drop(columns=['Loan_Status'], axis=1)
y = df_train['Loan_Status']

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)

### shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

### __________________________________________________________________________________________________________________________________________
### __________________________________________________________________________________________________________________________________________

## Model Training 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
def classify(model, X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Fit the model
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = model.score(X_test, y_test)
    print("Accuracy on Test Set:", round(acc * 100, 2), "%")
    # Cross-validation score
    score = cross_val_score(model, X, y, cv=5)
    print("Cross Validation Accuracy:", round(np.mean(score) * 100, 2), "%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n Confusion Matrix:")
    print(cm)

    # Accuracy Score (redundant but explicit)
    ac = accuracy_score(y_test, y_pred)
    print("Accuracy :",ac)
    # print("Accuracy Score:", round(ac * 100, 2), "%")

    # Bias and Variance
    bias = model.score(X_train, y_train)
    print("Bias :",bias)
    variance = model.score(X_test, y_test)
    print("variance :",variance)
    # print("Bias (Train Score):", round(bias * 100, 2), "%")
    # print("Variance (Test Score):", round(variance * 100, 2), "%")

### __________________________________________________________________________________________________________________________________________
### __________________________________________________________________________________________________________________________________________
## Check And run Model
from sklearn.svm import SVC
classifier = SVC()
classify(classifier, X, y)

from sklearn.svm import SVC
classifier = SVC()
classify(classifier, X, y)

from sklearn.linear_model import LogisticRegression
logreg  = LogisticRegression()
classify(logreg, X, y)

from sklearn.tree import DecisionTreeClassifier
dtree  = DecisionTreeClassifier()
classify(dtree , X, y)

from xgboost import XGBClassifier 
xgb = XGBClassifier()
classify(xgb, X, y)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
rfc = RandomForestClassifier()
classify(rfc, X, y)

## Hyperparameter Tunning 
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression(penalty='l1',    solver='liblinear',    dual=False,    tol=1e-4,
    C=100,    class_weight='balanced',    random_state=None,    max_iter=500 )
classify(lgr, X, y)

## Which Model Is Best?
#  **Best Overall Model: Logistic Regression**
# - Highest Test Accuracy: **83.74%**
# - Cross-validation is also consistent: **81.11%**
# - Bias and variance are **balanced** (low overfitting).

## Confussion Matrix 
model = LogisticRegression()
model.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
cm = confusion_matrix (y_test, y_pred)
print("confusion_matrix : ", cm)

## Predicting on the Test Dataset 
y_pred = model.predict(df_test)
df_test['Loan_Status'] = y_pred
df_test.head()
df_test['Loan_Status'] = df_test['Loan_Status'].map({1: 'Y', 0: 'N'})
df_test.head()