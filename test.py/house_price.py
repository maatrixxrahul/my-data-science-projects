import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

#create dataset
size = np.random.randint(500, 3000, 200)
bedrooms = np.random.randint(1, 6, 200)
age = np.random.randint(1, 50, 200)

#price formula
price = (size * 150) + (bedrooms * 10000)-(age * 500) + np.random.normal(0, 20000, 200)

df = pd.DataFrame({
    'Size_sqft': size,
    'Bedrooms': bedrooms,
    'Age_years': age,
    'Price': price.astype(int)
})
print(df.head())
print("n\Correlation with Price:")
print(df.corr()['Price'].sort_values(ascending=False))
X = df[['Size_sqft', 'Bedrooms', 'Age_years']]
y = df['Price']
#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#create model & test model
model = LinearRegression()
model.fit(X_train, y_train)
#predictions
predictions = model.predict(X_test)
# evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Mean Absolute error:", round(mae, 2))
print("r2 Score:", round(r2, 4))
# see coeficients of model
print("n\Model learned what:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {round(coef, 2)}")
print(f" Intercept: {round(model.intercept_, 2)}")
# model that not seen yet
new_house  = pd.DataFrame({
    'Size_sqft': [2000],
    'Bedrooms': [3],
    'Age_years': [10]
})
predicted_price = model.predict(new_house)
print(f"\npredicted price of new house: ${round(predicted_price[0], 2)}")

# verify using manual calculation
manual = (2000 * 150) + (3 * 10000) - (10 * 500)
print(f"price using manual formula: ${manual}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color='red', linewidth=2, label='Perfect Prediction Line')
plt.xlabel('Actual Price')
plt.ylabel('Prediced price')
plt.title('Actual vs Predicted house price')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.show()