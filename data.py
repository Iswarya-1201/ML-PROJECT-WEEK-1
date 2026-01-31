import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("global_sports_footwear_sales_2018_2026.csv")


X = df[['base_price_usd', 'units_sold']]
y = df['revenue_usd']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

m1, m2 = model.coef_
c = model.intercept_

print(f"Linear Regression Equation:")
print(f"Revenue = {m1:.2f} × Base_Price + {m2:.2f} × Units_Sold + {c:.2f}")


base_price = float(input("Enter base price (USD): "))
units = int(input("Enter units sold: "))


user_input = pd.DataFrame({
    'base_price_usd': [base_price],
    'units_sold': [units]
})

predicted_revenue = model.predict(user_input)

print(f"Predicted Revenue: ${predicted_revenue[0]:.2f}")
