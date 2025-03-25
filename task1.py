import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

Data=pd.read_csv('titanic.csv')

select_features=['Pclass','Sex','Age','SibSp','Parch',"Fare",'Embarked']
Data=Data[select_features+['Survived']]

Data['Age'].fillna(Data['Age'].median(),inplace=True)
Data['Embarked'].fillna(Data['Embarked'].mode()[0],inplace=True)

encode=LabelEncoder()
Data['Sex']=encode.fit_transform(Data['Sex'])
Data['Embarked']=encode.fit_transform(Data['Embarked'])

X = Data.drop(columns=["Survived"])
Y = Data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_predicted=model.predict(X_test)
Accuracy=accuracy_score(y_test,y_predicted)

print(f"Accuracy: {Accuracy:.2f}")
print(classification_report(y_test,y_predicted))

