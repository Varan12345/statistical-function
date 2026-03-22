import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\AMD\Downloads\archive (2)\Diwali Sales Data.csv", encoding='latin1')

print(df.head())
print(df.describe())    
print(df.info())
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x='Age Group', y='Amount', data=df)
plt.title('Amount Spent by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Amount Spent')
plt.show()


plt.pie(df['Age Group'].value_counts(), labels=df['Age Group'].value_counts().index, autopct='%1.1f%%')
plt.title('Distribution of Age Groups')
plt.show()

df.head()
print(df['Amount'].mean())
print(df['Amount'].median())   
print(df['Amount'].mode()[0])
print(df['Amount'].std())
print(df['Amount'].var())
