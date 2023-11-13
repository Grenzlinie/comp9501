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

cos_sim = cosine_similarity([original_data], [predicted_value])
plt.figure(figsize=(10, 7))
plt.plot(cos_sim[0])
plt.title('余弦相似度')
plt.xlabel('索引')
plt.ylabel('相似度')
plt.show()
