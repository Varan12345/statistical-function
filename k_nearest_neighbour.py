#k nearest neighbour 
#you are the average of the five people you spend the most time with-jim rohn
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

df= pd.read_csv(r"C:\Users\AMD\Documents\data.csv", encoding='utf-8')

print(df.columns)
df.drop(columns=['id','Unnamed: 32'],inplace=True,errors='ignore')
print(df.head())
print(df.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:],df.iloc[:,0] , test_size=0.2, random_state=2)

print(X_train.head())
print(X_train.shape)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
 
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
y_pred=knn.predict(X_test) 
print(accuracy_score(y_test,y_pred))

from sklearn.metrics  import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print (cm)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
 
import matplotlib.pyplot as plt 
k_values=[]
scores=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    k_values.append(k)
    scores.append(accuracy_score(y_test,y_pred))

plt.plot(k_values,scores)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.title("k vs acuracy")
plt.show()

X=df[['radius_mean', 'texture_mean']]
y=df['diagnosis']

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.neighbors import KNeighborsClassifier
 
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

import matplotlib.pyplot as plt
plt.scatter(X['radius_mean'],X['texture_mean'],c=y)
plt.xlabel("radius")
plt.ylabel("texture")
plt.show()