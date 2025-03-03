import numpy as np
import matplotlib.pyplot as plt
import make_classification as mc

X_train, X_test, y_train, y_test, a = mc.make_classification(d=2, n=100, u=10, seed=42)

# Visualization for 2D case
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", marker="o", label="Train Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", marker="s", label="Test Data")

# Plot decision boundary (hyperplane)
x_vals = np.linspace(-10, 10, 100)
y_vals = - (a[0] / a[1]) * x_vals  # Rearranging a[0]x + a[1]y = 0 to solve for y

plt.plot(x_vals, y_vals, "k--", label="Decision Boundary")
plt.axhline(0, color='gray', linestyle=':')
plt.axvline(0, color='gray', linestyle=':')
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Generated Linearly Separable Data")
plt.show()