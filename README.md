![image](https://github.com/user-attachments/assets/2ca8a5dc-eb01-4f0b-9ecd-4ca780ec2e21)# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the necessary python libraries to predict the employee churn rate.
2. Use pd.read_csv to deploy dataset. 
3. Declare x variable as input features except Departments as it have categorical values and use x.head to display the information of x.
4. Declare y as predicted variable to predict feaure name left y.head to display the information of y.
5. Use decisiontreeClassifier to classify the features as variable as dt.
6. Use fit to train the model before use train_test_split for training and test data.
7. Use predict for prediction and store in y_pred.
8. Use plt_plot to construct decision tree.
9. Use plt.show() to show the construction of decision tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VESHWANTH.
RegisterNumber: 212224230300
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
features=["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]
x=data[features]
x.head()
y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
![Screenshot 2025-04-23 091323](https://github.com/user-attachments/assets/f8add760-3a29-4714-b02d-b084af005354)
![Screenshot 2025-04-23 091404](https://github.com/user-attachments/assets/4459f586-53e6-4fd6-ba70-15a4db1ce6c6)
![Screenshot 2025-04-23 091436](https://github.com/user-attachments/assets/123c87bc-ebdc-4c0e-93c7-928ca8024a42)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
