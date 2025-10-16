# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries for data handling, modeling, and evaluation.

2.Load the Iris dataset using load_iris().

3.Convert data to DataFrame and add the target column.

4.Split data into features (X) and target (y).

5.Split X and y into training and testing sets using train_test_split().

6.Initialize the SGDClassifier with specified parameters.

7.Train the model on the training data.

8.Predict the target values for the test data.

9.Evaluate the model using accuracy score, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: DINESH S
RegisterNumber:  212224230069
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print("AADHITHYAA L")
print("21222422003")
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("AADHITHYAA L")
print("21222422003")
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("AADHITHYAA L")
print("21222422003")
print("Confusion Matrix:")
print(cm)

classification_report1 = classification_report(y_test,y_pred)
print("AADHITHYAA L")
print("21222422003")
print(classification_report1)
*/
```

## Output:
<img width="1720" height="730" alt="Screenshot 2025-09-25 105359" src="https://github.com/user-attachments/assets/d0aa854d-f769-4283-8fe4-e964d7d3b86c" />

____________________________________________________________________________________

<img width="393" height="66" alt="Screenshot 2025-09-25 105414" src="https://github.com/user-attachments/assets/98fffecf-d1eb-429e-9456-992b47198f4e" />

____________________________________________________________________________________

<img width="1288" height="388" alt="Screenshot 2025-09-25 105430" src="https://github.com/user-attachments/assets/b9ad05c4-3a13-44c4-8485-f170082ebfcd" />

____________________________________________________________________________________

<img width="1719" height="612" alt="Screenshot 2025-09-25 105452" src="https://github.com/user-attachments/assets/6554326b-1566-4c13-826e-9ebb46a9b384" />

____________________________________________________________________________________


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
