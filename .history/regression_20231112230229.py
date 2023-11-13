from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

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
