from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load data
data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl')
X = data.iloc[:, 1:7]
y = data.iloc[:, 7]

# 分层抽样
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

# create model
model = XGBRegressor()

# train model
model.fit(X_train, y_train)

# evaluate model
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

