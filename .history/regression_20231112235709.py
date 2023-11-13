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


# evaluate model
y_pred1 = model.predict(X_train)
y_pred2 = model.predict(X_test)

print('R2 Score:', r2_score(y_train, y_pred1))
print('R2 Score:', r2_score(y_test, y_pred2))

plt.scatter(y_train, y_pred1)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost Regression_train_data')
plt.show()


# draw plot to show the performance
plt.scatter(y_test, y_pred2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost Regression_test_data')
plt.show()

# save model
joblib.dump(model, 'Results/xgb_model.pkl')

