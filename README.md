# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values from dataframe and apply label encoder.
3. Apply decision tree classifier on the dataframe.
4. obtain the value of accuracy and data prediction
## Program:

~~~

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Varsha Ajith
RegisterNumber:  212221230118


import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
~~~

## Output:
## dataframe
![ex51](https://user-images.githubusercontent.com/94222288/203816052-61af543e-d600-45bf-9881-7f0f8269725e.png)
## null values
![ex52](https://user-images.githubusercontent.com/94222288/203816137-37dae550-911c-4c6b-b638-b76a219e2aa0.png)
![exp53](https://user-images.githubusercontent.com/94222288/203816204-81bb7ca4-c23b-4b32-8cdf-ac9d8209621a.png)
## Label encoder
![6d4](https://user-images.githubusercontent.com/94222288/203816373-75e5fd54-4177-40e9-980a-aaea6fdf456d.png)
Accuracy with entropy:
![65](https://user-images.githubusercontent.com/94222288/204594652-5cba1bce-3103-4bc4-8e1b-ca9dc3053033.png)

![66](https://user-images.githubusercontent.com/94222288/204594694-d349d232-80ed-4b25-86b1-61718dbfaf7f.png)

![67](https://user-images.githubusercontent.com/94222288/204594716-b1f77dc8-7e57-4e05-92af-ae036a4f73ac.png)

![68](https://user-images.githubusercontent.com/94222288/204594760-368f24bf-c4a5-4f56-8dce-32bac2c4ff3c.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
