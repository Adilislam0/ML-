import numpy as np
import matplotlib.pyplot as plt

# For reproducibility (so results don't change every run)
np.random.seed(42)

# Step 1: Generate synthetic data
# X is a column vector with 100 random values between 0 and 2
X = 2 * np.random.rand(100, 1)
# y = 4 + 3*x + noise  (noise ~ N(0,1))
y = 4 + 3 * X + np.random.randn(100, 1)

# Step 2: Add bias term (X0 = 1) -> X_b will be shape (100, 2)
# Column 1 = ones (for theta0), column 2 = X (for theta1)
X_b = np.c_[np.ones((100, 1)), X]

# Step 3: Initialize parameters (theta0 and theta1) as a column vector shape (2,1)
theta = np.random.randn(2, 1)

# Hyperparameters
learning_rate = 0.01
iterations = 1000
m = len(X)  # number of training examples (100)

# Step 4: Gradient Descent
cost_history = []

for i in range(iterations):
    # PREDICTION: h(x) = X_b.dot(theta)
    # shape: (100,2) dot (2,1) -> (100,1)
    y_pred = X_b.dot(theta)

    # ERRORS: difference between predictions and true y
    error = y_pred - y  # shape (100,1)

    # GRADIENT: (1/m) * X_b^T * error -> shape (2,1)
    gradients = (1 / m) * X_b.T.dot(error)

    # UPDATE RULE: theta := theta - alpha * gradients
    theta = theta - learning_rate * gradients

    # COST FUNCTION (Mean Squared Error / 2m)
    cost = (1 / (2 * m)) * np.sum(np.square(error))
    cost_history.append(cost)

    # Optional: print progress every 100 iterations
    if (i + 1) % 100 == 0:
        print(f"Iteration {i+1}: Cost = {cost:.4f}")

print("Final parameters (theta0, theta1):")
print(theta)

# Step 5: Plot regression line (data and fitted line)
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data")
X_line = np.array([[1], [2]])  # x values for plotting line (1 and 2)
X_line_b = np.c_[np.ones((2, 1)), X_line]
y_line = X_line_b.dot(theta)
plt.plot(X_line, y_line, "r-", linewidth=2, label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()

# Step 6: Plot cost function over iterations
plt.figure(figsize=(8, 4))
plt.plot(cost_history)
plt.title("Cost Function Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost (1/(2m) * sum squared error)")
plt.grid(True)
plt.show()
