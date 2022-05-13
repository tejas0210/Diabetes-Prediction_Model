# Import lib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load Data
data = pd.read_csv("diabetes1.csv")
print("\nOriginal Data :\n",data.head())

# Understand Data
print("\nUnderstanding Data :\n",data.isnull().sum())

# Features and Target
features = data[["FS","FU"]]
target = data["Diabetes"]

# handle categorical Data
nfeatures = pd.get_dummies(features, drop_first=True) 

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(nfeatures,target)

# Model
model1 = LogisticRegression()
model1.fit(x_train,y_train)
y_pred = model1.predict(x_test)
cr = classification_report(y_test,y_pred)
print("\nLogistic Regression Model : \n",cr)

model2 = DecisionTreeClassifier(criterion="gini")
model2.fit(x_train,y_train)
y_pred = model2.predict(x_test)
cr = classification_report(y_test,y_pred)
print("\nDecision Tree Classifier Model : \n",cr)

model3 = RandomForestClassifier(n_estimators=10)
model3.fit(x_train,y_train)
y_pred = model2.predict(x_test)
cr = classification_report(y_test,y_pred)
print("\nRandom Forest Classifier Model : \n",cr)

# Save Model
with open("db.model","wb") as f :
	pickle.dump(model3,f)