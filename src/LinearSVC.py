import numpy as np

class LinearSVC:


    def __init__(self, learning_rate = 0.01, epochs = 1000, reg_param = 0.01, random_seed = 1):


        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_param = reg_param
        self.random_seed = random_seed
        self.w = None
        self.b = None


    def net_input(self, X):

        return np.dot(X, self.w) + self.b


    def predict(self, X):

        return np.where(self.net_input(X) >= 0, 1, -1)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights and biases
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):

                margin = y[idx] * self.net_input(x_i)

                if margin < 1:
                    # misclassified or is within the margin

                    # gradient for weights includes regularization term and hinge loss derivative
                    self.w = self.w - self.learning_rate * (self.reg_param * self.w - y[idx] * x_i)

                    # derivative of loss w.r.t. b is -y if margin is < 1
                    self.b = self.b - self.learning_rate * (-y[idx])
                else:
                    self.w = self.w - self.learning_rate * (self.reg_param * self.w)

        return self


