import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

np.random
n = 1000

#create dataset - Telecom customer data
tenure = np.random.randint(1, 72, n) # joining tenure in months
monthly_charges = np.random.uniform(20, 100, n) # how much customer pays per month
total_charges = tenure * monthly_charges # total charges paid by customer
num_products = np.random.randint(1, 5, n) # number of products customer has
has_internet = np.random.randint(0, 2, n) # whether customer has internet service (0/1)
has_support = np.random.randint(0, 2, n) # whether customer has support service (0/1)
num_complaints = np.random.randint(0, 10, n) # number of complaints made by customer

#churn logic
#new customer + more complaints + lesss  products = high churn chances
churn_prob = (
    0.3 * (tenure < 12).astype(int) +
    0.25 * (num_complaints > 5).astype(int) +
    0.2 * (monthly_charges > 75).astype(int) +
    0.15 * (num_products == 1).astype(int) +
    0.1 * (has_support == 0).astype(int)
)
churn  = (churn_prob + np.random.normal(0, 0.1, n) > 0.4).astype(int) 

df = pd.DataFrame({
    'Tenure': tenure,
    'Monthly_Charges': monthly_charges.round(2),
    'Total_Charges': total_charges.round(2),
    'Num_Products': num_products,
    'Has_Internet': has_internet,
    'Has_Support': has_support,
    'Num_Complaints': num_complaints,
    'Churn': churn
})
print(df.head())
print(f"\nChurn rate: {round(df['Churn'].mean() * 100, 1)}%")
print(f"Dataset shape: {df.shape}")
# features and targets
X = df.drop('Churn', axis=1)
y = df['Churn']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 # train three models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)   

#compare accuracy
print("=" * 40)
print(f"{'Model':<20}{'Accuracy':>10}")
print("-" * 40)
print(f"{'Random Forest':<20}{accuracy_score(y_test, rf.predict(X_test)) * 100:>4f}%")
print(f"{'XGBoost':<20}{accuracy_score(y_test, xgb.predict(X_test)):>10.4f}")
print("=" * 40)
#feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance (XGBoost):")
print(feature_importance)
 #feature importance chart
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance -  customrer Churn(XGBoost)')
plt.tight_layout()
plt.savefig('churn_feature_importance.png')
plt.show()

