from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('./Data_Warehouse/data.csv')
X = data.iloc[:, 1:7]
y = data.iloc[:, 7]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
print('R2 Score:', r2_score(y_test, y_pred))

# Draw a picture to show the performance of model
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Linear Regression Model Performance')
plt.show()

