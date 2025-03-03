import numpy as np
import matplotlib.pyplot as plt
from make_classification import make_classification

from LinearSVC import LinearSVC

# d: number of features, n: total samples, u: uniform distribution range for features
d = 2
n = 500
u = 5

# Generate dataset
X_train, X_test, y_train, y_test, a = make_classification(d, n, u, seed=43)

# Initialize and train the LinearSVC
clf = LinearSVC(learning_rate=0.001, epochs=1000, reg_param=0.01, random_seed=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate test accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Optional: Plotting the decision boundary (only works for 2D data)
if d == 2:
    def plot_decision_boundary(X, y, model):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = model.predict(grid).reshape(xx.shape)

        plt.contourf(xx, yy, preds, alpha=0.3, levels=np.linspace(-1, 1, 3), cmap=plt.cm.Paired)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
        plt.title("Decision Boundary of LinearSVC")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    plot_decision_boundary(X_test, y_test, clf)
