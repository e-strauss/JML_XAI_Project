from sklearn.linear_model import lars_path
from sklearn.datasets import make_regression
import numpy as np

X = np.genfromtxt("data/lars_test_X.csv", delimiter=",", skip_header=1)
y = np.genfromtxt("data/lars_test_Y.csv", delimiter=",", skip_header=1)

print(X.shape, y.shape)

alphas, _, estimated_coef, its = lars_path(X, y, max_iter=3,return_n_iter=True, method='lasso')
print(its)
print(alphas)
print(estimated_coef)