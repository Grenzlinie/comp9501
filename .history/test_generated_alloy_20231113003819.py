import joblib
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = joblib.load('Results/xgb_model.pkl')
original_data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl').iloc[:, 7].values
generated_data = pd.read_excel('Results/generate_alloys.xlsx', engine='openpyxl')

predicted_value = model.predict(generated_data)

predicted_value_df = pd.DataFrame(predicted_value)
predicted_value_df.to_excel('Results/predicted_value.xlsx', index=False)

s = cosine_similarity([original_data], [predicted_value])
print(s)