from sklearn import datasets
from sklearn.model_selection import train_test_split

# define helper functions to evaluate
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

# load dataset
X, y = dataset.make_regression(
    n_samples=1000, n_features=1, noise=20, random_state=1
)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# instantiate regressor and fit
linreg = LinearRegression(learning_rate=0.01, n_iters=1000)
linreg.fit(X_train, y_train)

# make prediction
predictions = linreg.predict(X_test)
print(f"RMSE: {rmse(y_test, predicitions)}")