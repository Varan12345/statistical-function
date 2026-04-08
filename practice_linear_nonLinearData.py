import numpy as np
import matplotlib.pyplot as plt 
# from sklearn.linear_model import LogisticRegression

# X=np.array([[1],[2],[3],[4],[5],[6]])
# y=np.array([0,0,0,1,1,1])

# model=LogisticRegression()
# model.fit(X,y)

# plt.scatter(X,y,c=y)
# plt.xlabel('featrure')
# plt.ylabel('class')
# plt.title('linear data')
# plt.show()

from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_circles
X,y=make_circles(n_samples=200,noise=0.1)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y)
plt.title("non linear data")
plt.show()