import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ar= np.array([4,5,6,2,1,8,5,6,4,7])
# print(np.sum(ar))
# print (np.sort(ar))
# print(np.mean(ar))

dataset=pd.read_csv('C:/Users/AMD/Videos/Titanic-Dataset.csv')
print (dataset)
mn=np.mean(dataset["Age"])
print (mn)
sns.histplot(x='Age',data=dataset,bins=[i for i in range(0,81,10)])
plt.plot([mn for i in range(0,300)], [i for i in range(0,300)], c="red")
plt.show()
