import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the model
model = joblib.load('Results/xgb_model.pkl')
data = pd.read_excel('Results/generate_alloys.xlsx', engine='openpyxl')
X = data.iloc[:, 1:7]
y = data.iloc[:, 7]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)