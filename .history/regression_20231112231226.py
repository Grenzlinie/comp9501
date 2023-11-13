# Import necessary modules
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load data
data = pd.read_excel('Data_Warehouse/data.xlsx', engine='openpyxl')
X = data.iloc[:, 1:7]
y = data.iloc[:, 7]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert data to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float)
X_test = torch.tensor(X_test.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)

# Create a model
model = nn.Sequential(
    nn.Linear(6, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

# Evaluate the model
model.eval()
y_pred = model(X_test)
loss = criterion(y_pred, y_test)
print('Loss:', loss.item())

# Draw a picture to show the performance of model in test set
plt.scatter(y_test.detach().numpy(), y_pred.detach().numpy())
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Neural Network Model Performance')
plt.show()

