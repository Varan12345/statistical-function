import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import  load_iris

iris=load_iris()

print(iris)

# print(iris.data)

print(iris.target)


import seaborn as sns
df=sns.load_dataset('iris')
print (df.head())


#independet features and dependent feature
X=df.iloc[:,:-1]
y= iris.target

# split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.tree import DecisionTreeClassifier
# postpruning
treemodel=DecisionTreeClassifier(max_depth=2)

treemodel.fit(X_train,y_train)

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)
plt.show()

y_pred=treemodel.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score , classification_report
score=accuracy_score(y_pred,y_test)
print (score)

print(classification_report(y_pred,y_test))
