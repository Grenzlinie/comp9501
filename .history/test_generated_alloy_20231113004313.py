import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = joblib.load('Results/xgb_model.pkl')
original_data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl').iloc[:, 7].values
generated_data = pd.read_excel('Results/generate_alloys.xlsx', engine='openpyxl')

predicted_value = model.predict(generated_data)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(original_data, predicted_value, color='blue')
plt.xlabel('Original Data')
plt.ylabel('Predicted Value')
plt.title('Data Distribution')
plt.savefig('Results/data_distribution.png')
plt.close()

s = cosine_similarity([original_data], [predicted_value])
print(s)