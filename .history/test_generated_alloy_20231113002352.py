import joblib
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('Results/xgb_model.pkl')
original_data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl').iloc[:, 7].values
generated_data = pd.read_excel('Results/generate_alloys.xlsx', engine='openpyxl')

# plt.scatter(y_train, y_pred1, label='R2 Score: {:.2f}'.format(r2_train))
plt.xlabel('Real Value')
plt.ylabel('Predict Value')
plt.title('XGBoost_Train Data')
plt.legend()
plt.show()
# predicted_value = model.predict(generated_data)
# print('Predicted Value:', predicted_value)
# r2_generated_original = r2_score(original_data, predicted_value)
# print('R2 Score:', r2_generated_original)