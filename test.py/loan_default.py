import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import seaborn as sns
from sympy import rf

np.random.seed
n = 1000
#make dataset 
age = np.random.randint(22, 60, n)
income = np.random.randint(20000, 150000, n)
loan_amount = np.random.randint(5000, 50000 ,n)
credit_score = np.random.randint(300, 850, n)
months_employed = np.random.randint(0, 240, n)
#default probability
default_prob = (
    0.4 * (credit_score < 500).astype(int) +
    0.3 * (income < 40000).astype(int) +
    0.2 * (loan_amount > 30000).astype(int) +
    0.1 * (months_employed < 12).astype(int) 
)
default = (default_prob + np.random.normal(0, 0.1, n) > 0.4).astype(int)

df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Loan_Amount': loan_amount,
    'Credit_Score': credit_score,
    'Months_Employed': months_employed,
    'Default': default 
})
print(df.head())
print("\nDefault distribution:")
print(df['Default'].value_counts())
print(f"\nDefault rate: {round(df['Default'].mean() * 100, 1)}%")
X = df[['Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Months_Employed']]
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3 models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}
# compare and train three models
print("=" * 45)
print(f"{'Model':<25} {'Accuracy':>10}")
print("=" * 45)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name:<25} {acc:>10.4f}")
    print("=" * 45)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt= 'd', cmap= 'Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Deafult', 'Deafault'])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('predicted')
plt.savefig('confusion_matrix.png')
plt.show()

print("n\Classification Report:")
print(classification_report(y_test, rf_pred,
                            target_names=['No Default', 'Default']))