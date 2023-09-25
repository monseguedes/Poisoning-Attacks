from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# prepare dataset
# housing = fetch_california_housing()
diabetes = datasets.load_diabetes()
X = diabetes.data
X = X[:, 2]  # use only one feature
y = diabetes.target

# # Plot the third feature against the target
# plt.scatter(X, y)
# plt.show()

# # Plot the same but with normalized intercept
# plt.scatter(X, y - np.mean(y))
# plt.show()

# add the w_0 intercept where the corresponding x_0 = 1
# Xp = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

# fit the model without regularizing intercept in sklearn
alpha = 0.5
ridge = Ridge(fit_intercept=True, alpha=alpha)
X = X.reshape(-1, 1)
ridge.fit(X, y)

# # Plot data with fitted regression line
# plt.scatter(X, y)
# plt.plot(X, ridge.predict(X), color='red')
# # Add mean of y 
# plt.plot(X, np.mean(y)*np.ones(X.shape), color='green')
# plt.show()


# Try to reproduce sklearn with closed form.
weights = np.linalg.inv(X.T.dot(X) + alpha * np.eye(X.shape[1])).dot(X.T).dot(y - np.mean(y)) 
# intercept= np.mean(y) - weights.dot(np.mean(X))
intercept = np.mean(y)

# # Plot data with fitted regression line
# plt.scatter(X, y)
# plt.plot(X, weights*X + intercept, color='red')
# # Add mean of y
# plt.plot(X, np.mean(y)*np.ones(X.shape), color='green')
# plt.show()

# Combine both plots in one figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Ridge Regression')
ax1.scatter(X, y)
ax1.plot(X, ridge.predict(X), color='red')
ax1.plot(X, np.mean(y)*np.ones(X.shape), color='green')
ax1.set_title('sklearn')
ax2.scatter(X, y)
ax2.plot(X, weights*X + intercept, color='red')
ax2.plot(X, np.mean(y)*np.ones(X.shape), color='green')
ax2.set_title('closed form')
plt.show()