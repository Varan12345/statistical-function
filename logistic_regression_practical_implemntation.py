import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
   
columns=['sepal_length','sepal_width','petal_length','petal_width','species']

df=pd.read_csv(r"C:\Users\AMD\Documents\iris.data.csv", names=columns, encoding="latin1")

print(df.head())
print(df.columns)
# print(df['species'].unique())
print(df.isnull().sum())
df['species']=df['species'].str.strip().str.lower()
df['species']=df['species'].str.replace('iris-','')
print(df['species'].unique())
print(df[df['species']!='setosa'])
print(df.head())
df['species']=df['species'].map({'versicolor':0,'virginica':1})
print(df.head())
## split this datset into indepndent amd dependent features

from sklearn.preprocessing import LabelEncoder

# X and y
X = df.drop('species', axis=1)

le = LabelEncoder()
y = le.fit_transform(df['species'])

# split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# grid
from sklearn.model_selection import GridSearchCV
parameter = {
    'penalty': ['l2'],
    'C': [1,2,3,4,5],
    'max_iter': [100,200]
}

grid = GridSearchCV(classifier, param_grid=parameter, scoring='accuracy', cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

y_pred =grid.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report

score=accuracy_score(y_pred,y_test)
print(score)

sns.pairplot(df,hue='species')
plt.show()