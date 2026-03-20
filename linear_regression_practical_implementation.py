import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

dataset = df
print(dataset.head())

#independent feateure and dependent feature

# independent features
X = dataset.drop("medv", axis=1)

# dependent feature
y = dataset["medv"]

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

## standardizing the dataset

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#scaler.inverse_transform(X_train)
         

from sklearn.linear_model import LinearRegression           
from sklearn.model_selection import cross_val_score
regression=LinearRegression()
regression.fit(X_train,y_train)
mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=5)      
np.mean(mse)

##prediction

reg_predict=regression.predict(X_test)
print (reg_predict)

import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(reg_predict-y_test,kind="kde")
plt.show()

from sklearn.metrics import r2_score
score=r2_score(reg_predict,y_test)
print (score)
print(df.mean())
print(df.std())
print(df.corr())