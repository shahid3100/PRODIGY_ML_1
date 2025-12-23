import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("house_data.csv")

# Features and target
X = data[['Size_sqft', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("Actual Prices:", y_test.values)
print("Predicted Prices:", predictions)

# Predict new house price
new_house = [[1600, 3, 2]]
price = model.predict(new_house)
print("Predicted price for new house:", price[0])

# Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
