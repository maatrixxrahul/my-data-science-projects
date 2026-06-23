import pandas as pd
import numpy as np
np.random.seed(42)
data = {'date': pd.date_range(start='2024-01-01', periods=200,freq='D'),
        'product': np.random.choice(['Laptop', 'Mobile','Tablet', 'Headphone'], 200),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 200),
        'Quantity': np.random.randint(1, 20, 200),
        'Price': np.random.choice([500, 1000, 1500, 2000,250], 200)
}
df = pd.DataFrame(data)
df['Sales'] = df['Quantity'] * df['Price']
df.to_csv('/Users/macbook/CLV prediction/sales_data.csv', index=False)
print(df.head(10))
print(df.shape)
region_sales = df.groupby('Region')['Sales'].sum()
print(region_sales)
product_sales = df.groupby('product')['Sales'].sum()
print(product_sales)
region_avg = df.groupby('Region')['Sales'].mean()
print(region_avg)
combo = df.groupby(['Region', 'product'])['Sales'].sum()
print(combo)
import matplotlib.pyplot as plt
region_sales = df.groupby('Region')['Sales'].sum()
plt.figure(figsize=(8, 5))
plt.bar(region_sales.index, region_sales.values, color='skyblue')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.savefig('/Users/macbook/CLV prediction/region_sales.png')
plt.show()
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.barplot(x='Region', y='Sales', data=df, estimator=sum, errorbar=None)
plt.title('Total Sales by Region(Seaborn)')
plt.savefig('region_sales_seaborn.png')
plt.show()
df['date'] = pd.to_datetime(df['date'])
daily_sales = df.groupby('date')['Sales'].sum()

plt.figure(figsize=(10, 6))
plt.plot(daily_sales.index, daily_sales.values, color='blue')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/macbook/CLV prediction/Sales_Trend.png')
plt.show()
