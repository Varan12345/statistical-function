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
# df['species']=df['species'].map({'versicolor':0,'virginica':1})
print(df.head())
## split this datset into indepndent amd dependent features

from sklearn.preprocessing import LabelEncoder
df=df[df['species']!='setosa']
df['species']=df['species'].map({'versicolor':0,'virginica':1})
# X and y
X = df.drop('species', axis=1)

y=df['species']

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

score=accuracy_score(y_test,y_pred)
print(score)




sns.pairplot(df,hue='species')
plt.show()
# confusion materices  
# actual | predited
# 1|1        true postive(tp)
# 0|0        true negative(tn)
# 0|1        false postive(fp)
# 1|0        false negative(fn)

# accuracy=is the ration of corretly predictd observation to the total observation
#  accuracy =tp+tn/tp+tn+fp+fn
# prcesion= is the ration of correct observation to the total prediction post+ observation
#  precision=tp/tp+fp
# recall= is the ration of correctly predicted observation to all postive observation
# recall =tp/tp+fn 

# f beta score (1+b^2) precison*recall/b^2[precison+recall]
# fp&fn both are imp. and when we both have to perform precison and recll(b=1)//hatmonic mean
#  fp>fn fp is more imp than fn (b=0.5)
# fn>>fp (b=2)
#  f1 score is the harmonic mean of precison recall

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# prediction already hai
# y_test = actual
# y_pred = model prediction

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Precision
prec = precision_score(y_test, y_pred)
print("Precision:", prec)

# Recall
rec = recall_score(y_test, y_pred)
print("Recall:", rec)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

for actual, pred in zip(y_test,y_pred):print(actual, pred )

tp=tn=fp=fn=0
for actual, pred in zip(y_test,y_pred): 
    if actual==1 and pred==1:
        tp+=1
    elif actual==0 and pred==0:
     tn+=1
    elif actual==0 and pred==1:
      fp+=1
    elif actual==1 and pred==0:
      fn+=1

      print("TP:",tp)
      print("TN:",tn)
      print("FP:", fp)
      print("FN:", fn)