import numpy as np 

class LinearRegression:
    def __init__(self, learning_rate = 1e-3, n_iters = 1000):
        # init parameters
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def _init_params(self):
        self.weights = np.zeros(self.n_features)
        self.bias = 0
    
    def _update_params(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        
    
    def _get_prediction(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def _get_gradients(self, X, y, y_pred):
        # get distance between y_pred and y_true
        error = y_pred - y_true
        # compute the gradients of weight and bias
        dw = (1/self.n_samples) * np.dot(X.T, error)
        db = (1/self.n_samples) * np.sum(error)
        return dw, db
    
    def fit(self, X, y):
        # get number of samples and features
        self.n_samples, self.n_features = X.shape
        # init weights and bias
        self.init_params
        
    # perform gradient descent for n iterations (essentially this is the gradient descent algorithm)
    for i in range(self.n_iters):
        # get y prediction
        y_pred = self.get_prediction(X)
        # compute gradients
        dw, db = self.get_gradients(X, y, y_pred)
        # update weights and bias with gradients
        self._update_params(dw, db)
    
    def predict(self, X):
        y_pred = self.get_prediction(X)
        return y_pred
    
    
    