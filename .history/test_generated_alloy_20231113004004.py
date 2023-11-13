import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = joblib.load('Results/xgb_model.pkl')
original_data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl').iloc[:, 7].values
generated_data = pd.read_excel('Results/generate_alloys.xlsx', engine='openpyxl')

predicted_value = model.predict(generated_data)

s = cosine_similarity([original_data], [predicted_value])
print(s)