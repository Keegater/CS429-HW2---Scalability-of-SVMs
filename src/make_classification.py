import numpy as np
from sklearn.model_selection import train_test_split

def make_classification(d, n, u, seed=42):
    np.random.seed(seed)

    a = np.random.randn(d)

    X = np.random.uniform(-u, u, size=(n, d))

    y = np.sign(X @ a)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    return X_train, X_test, y_train, y_test, a