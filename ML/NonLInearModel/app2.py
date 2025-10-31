import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Seed
np.random.seed(123)
torch.manual_seed(123)

samples = abs(int(input('Type the amount of samples you want to generate: ')))
epochs = abs(int(input('Type the total amount of training epochs: ')))

# Data
x = np.random.randint(1, 101, size=samples).reshape(-1, 1)
y = 2 * x + np.random.randn(samples, 1)

# Independent scalers
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x = torch.from_numpy(x_scaler.fit_transform(x)).float()
y = torch.from_numpy(y_scaler.fit_transform(y)).float()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

# Model
model = nn.Sequential(
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
loss_crit = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for i in range(epochs + 1):
    y_pred = model(X_train)
    loss = loss_crit(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f'Epoch {i}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)

# Convert back to original scale
x_test_original = x_scaler.inverse_transform(X_test.numpy())
y_test_original = y_scaler.inverse_transform(y_test.numpy())
y_test_pred_original = y_scaler.inverse_transform(y_test_pred.numpy())

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x_test_original, y_test_original, label='Real Values', color='green', alpha=0.6)
plt.scatter(x_test_original, y_test_pred_original, label='Predicted Values', color='red', alpha=0.6)
plt.title('Equation Learning Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()