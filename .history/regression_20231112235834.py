from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import joblib

# Load data
data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl')
X = data.iloc[:, 1:7]
y = data.iloc[:, 7]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create model
model = XGBRegressor()
eval_set = [(X_train, y_train), (X_test, y_test)]
# train model
model.fit(X_train, y_train, eval_metric=["rmse"], eval_set=eval_set, verbose=False)
results = model.evals_result()

# 绘制损失曲线
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()


# 评估模型
y_pred1 = model.predict(X_train)
y_pred2 = model.predict(X_test)

r2_train = r2_score(y_train, y_pred1)
r2_test = r2_score(y_test, y_pred2)
print('R2 Score:', r2_train)
print('R2 Score:', r2_test)

# 绘制图表以显示性能
plt.scatter(y_train, y_pred1, label='R2 Score: {:.2f}'.format(r2_train))
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('XGBoost 回归_训练数据')
plt.legend()
plt.show()

plt.scatter(y_test, y_pred2, label='R2 Score: {:.2f}'.format(r2_test))
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('XGBoost 回归_测试数据')
plt.legend()
plt.show()

# save model
joblib.dump(model, 'Results/xgb_model.pkl')

