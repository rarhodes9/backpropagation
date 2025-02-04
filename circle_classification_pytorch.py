"""
PyTorch Implementation of a Simple Neural Network for Binary Classification

Step 2) Implement backpropogation algorithm with Pytorch using circle classification as previous file for consistency.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------
# 1. Generate Custom Dataset
# --------------------------------------------------
num_samples = 500
radius = 1.0

# Create a random number generator with no fixed seed
rng = np.random.default_rng()

# Generate random points in [-2, 2] x [-2, 2]
X = rng.uniform(-2, 2, (num_samples, 2))


# Calculate the distance of each point from the origin (0,0).
# Formula: distance = sqrt(x^2 + y^2)
distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)

# If distance > radius => outside the circle => label=1, else label=0
y = (distances > radius).astype(int)
T = y.reshape(-1, 1)

# --------------------------------------------------
# 2. Train/Test Split
# --------------------------------------------------
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2, random_state=None)
print(f"Train shapes: X={X_train.shape}, T={T_train.shape}")
print(f"Test shapes : X={X_test.shape}, T={T_test.shape}")

# Convert NumPy arrays to PyTorch tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
T_train_tensor = torch.tensor(T_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
T_test_tensor = torch.tensor(T_test, dtype=torch.float32)

# --------------------------------------------------
# 3. Define the Neural Network Model using PyTorch
# --------------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Xavier/Glorot initialization for weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = torch.sigmoid(self.fc1(x))
        # Output layer with sigmoid activation for binary classification
        x = torch.sigmoid(self.fc2(x))
        return x

# Hyperparameters
input_dim = 2
hidden_dim = 4
output_dim = 1
learning_rate = 0.5
epochs = 2000

# Instantiate the model, loss function and optimizer
model = SimpleNN(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# --------------------------------------------------
# 4. Train the Neural Network
# --------------------------------------------------
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, T_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 200 epochs on both train and test sets.
    if (epoch + 1) % 200 == 0:
        model.eval() 
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, T_test_tensor)
        print(f"Epoch {epoch+1:4d}, Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}")
        model.train()  # switch back to training mode

# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------
model.eval()
with torch.no_grad():
    train_preds = (model(X_train_tensor) >= 0.5).float()
    test_preds = (model(X_test_tensor) >= 0.5).float()

train_acc = (train_preds.eq(T_train_tensor).sum().item() / T_train_tensor.shape[0]) * 100
test_acc = (test_preds.eq(T_test_tensor).sum().item() / T_test_tensor.shape[0]) * 100

print(f"\nFinal Results:")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy:  {test_acc:.2f}%")

# --------------------------------------------------
# 6. Visualization of Decision Boundary with Test Data
# --------------------------------------------------
xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
with torch.no_grad():
    grid_preds = (model(grid_points_tensor) >= 0.5).float().numpy()
grid_preds = grid_preds.reshape(xx.shape)

# Use a discrete colormap: blue for class 0 (inside) and red for class 1 (outside)
cmap = ListedColormap(["blue", "red"])

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, grid_preds, alpha=0.3, cmap=cmap)
plt.scatter(X_test[:, 0], X_test[:, 1], c=T_test.ravel(), cmap=cmap, edgecolors='k', alpha=0.7)

# Draw the true circle boundary for reference.
circle = plt.Circle((0, 0), radius, color='black', fill=False, linewidth=2)
plt.gca().add_patch(circle)

# Legend handles
# Predicted regions (background) are shown as squares with transparency.
pred_inside = Line2D([], [], marker='s', color='w', markerfacecolor='blue',
                       markersize=15, alpha=0.3, label='Predicted: Inside (0)')
pred_outside = Line2D([], [], marker='s', color='w', markerfacecolor='red',
                        markersize=15, alpha=0.3, label='Predicted: Outside (1)')

# Actual outcomes are shown as circles.
actual_inside = Line2D([], [], marker='o', color='blue', markeredgecolor='k',
                       markersize=10, linestyle='None', label='Actual: Inside (0)')
actual_outside = Line2D([], [], marker='o', color='red', markeredgecolor='k',
                        markersize=10, linestyle='None', label='Actual: Outside (1)')

# Combine legend handles.
legend_handles = [pred_inside, pred_outside, actual_inside, actual_outside]
plt.legend(handles=legend_handles, loc="upper right")

plt.axis("equal")
plt.title("Decision Boundary")
plt.show()
