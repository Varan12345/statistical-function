import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ar= np.array([4,5,6,2,1,8,5,6,4,7])
# print(np.sum(ar))
# print (np.sort(ar))

data=[2,3,3,4,4,4,5,5,5,5,6,6,6,6,6,7,7,7,7,7,7,8,8,8,8,8,9,9,9,9,10,10,10,11,11,12]
df=pd.DataFrame({"x":data})
print(df["x"].skew())
print(df["x"].mean(),df["x"].median(),df["x"].mode())

sns.histplot(x='x',data=df,bins=[2,3,4,5,6,7,8,9,10,11,12,13])
plt.show()

dataset=pd.read_csv('C:/Users/AMD/Videos/Titanic-Dataset.csv')
print (dataset)
print(dataset.isnull().sum())
mn=np.mean(dataset["Age"])
print (mn)
kn=np.median(dataset['Fare'])
al=dataset["Fare"].mode()[0]
print(dataset['Fare'].value_counts())

sns.histplot(x='Age',data=dataset,bins=[i for i in range(0,81,10)])

plt.plot([mn for i in range(0,300)], [i for i in range(0,300)], c="red")
plt.plot([kn for i in range(0,300)], [i for i in range(0,300)], c="purple")  #median 
plt.plot([al for i in range(0,300)], [i for i in range(0,300)], c="green") #mode
plt.show()

# range=maximum-minimum
#  mean absolute error(MAD)=summison|xi-x^-|/n
min_r=dataset['Age'].min()
max_r=dataset['Age'].max()
range=max_r-min_r
print (range)

sec_a=np.array([75,65,73,68,72,76])
sec_b=np.array([90,47,43,96,93,51])
no=np.array([1,2,3,4,5,6])
mean=np.mean(sec_b)

mad_a=np.sum(np.abs(sec_a-mean))/len(sec_a)
mad_b=np.sum(np.abs(sec_b-mean))/len(sec_b)

print(mad_a,mad_b)

plt.figure(figsize=(10,3))
plt.scatter(sec_a,no,label="sec_a")
plt.scatter(sec_b,no,color='red',label="sec_b")
plt.plot([70,70,70,70,70,70],no,color="purple",label="mean")
plt.legend()
plt.show()


print (dataset["Age"].var())
print(dataset["Age"].std())
print(dataset.describe())
sns.histplot(x="Age",data=dataset)
plt.show()

print(np.percentile(dataset["Age"],25))

Skewness tells the shape of data distribution (left or right tilted)
📊 Types of Skewness (very important) curve
normal distribution(no skew)
2️⃣ Positive Skew (Right skew)
3️⃣ Negative Skew (Left skew)



print(dataset['Age'].skew())

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

data=sns.load_dataset('tips')
print(data.head(3))
print(data.info())


Correlation tells how strongly two variables are related (range: -1 to +1)
Covariance tells direction of relationship (positive or negative), but NOT strength clearly

data_corr=data.select_dtypes(["float64","int64"]).corr()
data_cov=data.select_dtypes(["float64","int64"]).cov()

sns.heatmap(data_corr,annot=True)
plt.show()
plt.figure(figsize=(7,6))
sns.heatmap(data_cov,annot=True)
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

# Population data
pop_data = [np.random.randint(10,100) for i in range(10000)]
pop_table = pd.DataFrame({"pop_data":pop_data})

print("Population Mean:", np.mean(pop_data))

# Population distribution
sns.kdeplot(x="pop_data", data=pop_table)
plt.title("Population Distribution")
plt.show()

# Sampling
sample_mean = []

for i in range(50):  # number of samples
    sample_data = np.random.choice(pop_data, size=500)
    sample_mean.append(np.mean(sample_data))

print("Mean of Sample Means:", np.mean(sample_mean))

# Sampling distribution
Sample_m = pd.DataFrame({"Sample_mean":sample_mean})

sns.kdeplot(x="Sample_mean", data=Sample_m)
plt.title("Sampling Distribution (CLT)")
plt.show()

hypothesis testing 
null hypothesis=hO(hnot)
alternative hypothesis (Ha)


z test  smaple data>30
pop std deviation
mean()
no of sample data
big data


t  test smaple data<30
population data mean()
sample data mean and std 
no of sample data
small data


find crtical values 
“We assume truth (H₀) and try to prove it wrong using data.”

a teacher claims that the mean score of student in his class is greater than 82 with a standard deaviatiom
of 20  if a smaple of 81 students was selected with a mean score of 90
(z test)
import scipy.stats as st
import numpy as np
s_x=90
p_u=82
p_std=20
n=81
ap=0.05

z_cal=(s_x-p_u)/(p_std/np.sqrt(n))
print (z_cal)
z_table=st.norm.ppf(1-ap)
print (z_table)

if z_table< z_cal:
    print("ha is right")

else:
    print("ho is right")

a manufactuer calims that the avg weight of a bag of potato is 150gm
a sample of 25 bags is taken and the avg weight is found to be 148grams
with a stadard deviation of of 5 grams test the manufacturer claims using a 
one tailed t test with a significance level of 0.05

import scipy.stats as st
t_table =st.t.ppf(0.05,24)
print (t_table)

import scipy.stats as st
import numpy as np
t_table =st.t.ppf(0.025,38)
print (t_table)

t_cal=(80-75)/np.sqrt((25/20)+(26/20))
print(t_cal)


chi-square test

import numpy as np

ob=np.array([22,17,20,26,22,13])
ex=np.array([20,20,20,20,20,20])
print(np.sum(np.square(ob-ex)/ex))


 
import numpy as np

row1 = np.array([40,45,25,10])
row2 = np.array([35,30,20,30])

# Total
total = np.sum(row1) + np.sum(row2)

# Row sums
sum_r1 = np.sum(row1)
sum_r2 = np.sum(row2)  
sum_row = np.array([sum_r1, sum_r2])

# Column sums
sum_col = row1 + row2

# Expected values
exp = []
for i in sum_row:
    for j in sum_col:
        exp.append((i*j)/total)

exp = np.array(exp)

# Observed values
obs = np.array([40,45,25,10,35,30,20,30])

# Chi-square formula
chi = np.sum(((obs - exp)**2) / exp)

print("Chi-square value:", chi)