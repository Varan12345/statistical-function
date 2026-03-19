import pandas as pd

df = pd.read_csv(r"C:\Users\AMD\Downloads\amazon.csv\amazon.csv", encoding='latin1')

df['discounted_price'] = df['discounted_price'].str.replace('[^0-9.]','', regex=True).astype(float)
df['actual_price'] = df['actual_price'].str.replace('[^0-9.]','', regex=True).astype(float)

print(df[['discounted_price','actual_price']].head())
print(df.describe())
print(df.info())

import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x='category', y='discounted_price', data=df)
plt.title('Discounted Price by Category')
plt.xlabel('Category')
plt.ylabel('Discounted Price')
plt.xticks(rotation=45)
plt.show()