"""

Step 1) Implement backpropogation algorithm from scratch

Plan: Generate points labeled by whether they're inside/outside a circle, build and train a 
simple neural network from scratch (NumPy) and visualize results. 


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

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

# --------------------------------------------------
# 3. Define the Neural Network Class
# --------------------------------------------------
class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=None):
        """
        Initialize network parameters using Xavier/Glorot initialization.
        """
        self.rng = np.random.default_rng(seed)
        
        # Xavier/Glorot initialization for W1:
        # W1 ~ N(0, 1) * sqrt(1 / input_dim)
        self.W1 = self.rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(1.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        # Xavier/Glorot initialization for W2:
        # W2 ~ N(0, 1) * sqrt(1 / hidden_dim)
        self.W2 = self.rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
    
    @staticmethod
    def sigmoid(z):
        """
        Numerically stable sigmoid activation function:
        sigmoid(z) = 1 / (1 + e^(-z))
        """
        return 1.0 / (1.0 + np.exp(-z))
    
    def forward(self, X):
        """
        Forward pass.
        
        Z1 = X * W1 + b1  
        A1 = sigmoid(Z1)
        
        Z2 = A1 * W2 + b2
        A2 = sigmoid(Z2)
        
        Returns:
            A1: Activation of the hidden layer.
            A2: Activation of the output layer 
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A1, self.A2
    
    def compute_loss(self, T):
        """
        Compute the Binary Cross-Entropy (BCE) loss.

        BCE because of Binary Classification
        
        BCE formula (for a single sample):
        -[ T * log(A2) + (1 - T) * log(1 - A2) ]
        
        For m samples:
        loss = -(1/m) * Σ [ T_i * log(A2_i) + (1 - T_i) * log(1 - A2_i) ].
        """
        m = T.shape[0]
        eps = 1e-8  # Small constant to prevent log(0)
        
        # Clip predictions to avoid log(0) (inf). Keeps eps between 0 and 1 - eps.
        A2_clipped = np.clip(self.A2, eps, 1 - eps)
        
        # Compute the BCE loss
        loss = -np.sum(T * np.log(A2_clipped) + (1 - T) * np.log(1 - A2_clipped)) / m
        return loss
    
    def backward(self, X, T):
        """
        Perform backpropagation to compute gradients.
        
        With BCE loss + sigmoid output:
        dZ2 = A2 - T
        
        Then propagate backwards:
        
        dW2 = (A1^T * dZ2) / m
        db2 = sum of dZ2 across samples / m
        
        dA1 = dZ2 * W2^T
        dZ1 = dA1 * (A1 * (1 - A1))   (derivative of sigmoid)
        
        dW1 = (X^T * dZ1) / m
        db1 = sum of dZ1 across samples / m
        """
        m = X.shape[0]
        
        # Gradient at output layer using derivative of BCE + sigmoid
        dZ2 = self.A2 - T
        
        # Gradients for W2 and b2
        self.dW2 = np.dot(self.A1.T, dZ2) / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backprop to hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        
        # Sigmoid derivative: dZ1 = dA1 * A1 * (1 - A1)
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        
        # Gradients for W1 and b1
        self.dW1 = np.dot(X.T, dZ1) / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    def update_parameters(self, learning_rate):
        """
        Update the network's parameters using gradient descent:
        
        W1 = W1 - η * dW1
        b1 = b1 - η * db1
        W2 = W2 - η * dW2
        b2 = b2 - η * db2
        
        where η is the learning_rate.
        """
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
    
    def predict(self, X):
        """
        Predict binary class labels for input data X.
        
        If A2 >= 0.5, class=1; else class=0.
        """
        _, A2 = self.forward(X)
        return (A2 >= 0.5).astype(int)

# --------------------------------------------------
# 4. Train the Neural Network
# --------------------------------------------------
input_dim = 2
hidden_dim = 4
output_dim = 1
learning_rate = 0.5
epochs = 2000

# Create the neural network instance without a fixed seed for weight initialization.
nn = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim, seed=None)

for epoch in range(epochs):
    # Forward pass on training data
    nn.forward(X_train)
    
    # Compute training loss (BCE)
    loss_train = nn.compute_loss(T_train)
    
    # Backprop to get gradients
    nn.backward(X_train, T_train)
    
    # Update parameters
    nn.update_parameters(learning_rate)
    
    # Print progress every 200 epochs
    if (epoch + 1) % 200 == 0:
        nn.forward(X_test)
        loss_test = nn.compute_loss(T_test)
        print(f"Epoch {epoch+1:4d}, Train Loss={loss_train:.4f}, Test Loss={loss_test:.4f}")

# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------
train_preds = nn.predict(X_train)
test_preds = nn.predict(X_test)

train_acc = np.mean(train_preds == T_train) * 100
test_acc = np.mean(test_preds == T_test) * 100

print(f"\nFinal Results:")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy:  {test_acc:.2f}%")

# --------------------------------------------------
# 6. Visualization of Decision Boundary with Test Data
# --------------------------------------------------
# Create a grid of points across [-2, 2] in both x and y dimensions
xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))

# Combine grid coordinates into a (200*200, 2) array
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict class (0 or 1) at each point in the grid
grid_preds = nn.predict(grid_points)

# Reshape the predictions back into (200, 200) for contour plotting
grid_preds = grid_preds.reshape(xx.shape)

# Use a discrete colormap: blue for class 0 (inside) and red for class 1 (outside)
cmap = ListedColormap(["blue", "red"])

plt.figure(figsize=(6, 6))

# Plot filled contours based on predicted labels
plt.contourf(xx, yy, grid_preds, alpha=0.3, cmap=cmap)

# Plot the test set points, colored by their true labels
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
