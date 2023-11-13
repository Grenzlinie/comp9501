import joblib
import pandas as pd

# Load the model
model = joblib.load('Results/xgb_model.pkl')
data = pd.read_excel('Results/generate_alloys.xlsx', engine='openpyxl')