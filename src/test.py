import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import make_classification as mc

u = 100
X_train, X_test, y_train, y_test, a = mc.make_classification(d=2, n=100, u=u, seed=13)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test) 

plt.figure(figsize=(12, 10))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", marker="o", label="Train Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_predict, cmap="coolwarm", marker="x", label="Test Data")

x_vals = np.linspace(-u, u, 100)
y_vals = - (clf.coef_[0][0] / clf.coef_[0][1]) * x_vals - (clf.intercept_[0] / clf.coef_[0][1])

plt.plot(x_vals, y_vals, "k--", label="Hyperplane (a)")
plt.axhline(0, color='gray', linestyle=':')
plt.axvline(0, color='gray', linestyle=':')
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Classification with Decision Boundary and True Hyperplane")
plt.show()
