import joblib
import pandas as pd
from sklearn.metrics import r2_score

# Load the model
model = joblib.load('Results/xgb_model.pkl')
original_data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl').iloc[:, 7].values
generated_data = pd.read_excel('Results/generate_alloys.xlsx', engine='openpyxl')

predicted_value = model.predict(generated_data)
r2_generated_original = r2_score(predicted_value, original_data)
print('R2 Score:', r2_generated_original)