from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 加载数据
data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl')
X = data.iloc[:, 1:7]
y = data.iloc[:, 7]

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型，增加正则化参数以防止过拟合
model = XGBRegressor(reg_alpha=0.5, reg_lambda=0.5)

# 训练模型
model.fit(X_train, y_train)

# 评估模型
y_pred1 = model.predict(X_train)
y_pred2 = model.predict(X_test)

print('R2 Score:', r2_score(y_train, y_pred1))
print('R2 Score:', r2_score(y_test, y_pred2))

plt.scatter(y_train, y_pred1)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost Regression')
plt.show()


# draw plot to show the performance
plt.scatter(y_test, y_pred2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost Regression')
plt.show()

