from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl')
X = data.iloc[:, 1:7]
y = data.iloc[:, 7]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# create model
model = XGBRegressor()

# train model
model.fit(X_train, y_train)

# evaluate model
y_pred1 = model.predict(X_train)
y_pred2 = model.predict(X_test)

from sklearn.metrics import r2_score
print('R2 Score:', r2_score(y_train, y_pred1))
print('R2 Score:', r2_score(y_test, y_pred2))

# 绘制图像以显示模型在测试集中的性能
plt.scatter(y_test, y_pred2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost Regression')
plt.show()

