import numpy as np

from econml.dml import DML, LinearDML
from sklearn.linear_model import LassoCV

from econml.inference import BootstrapInference

# Simulate data
np.random.seed(1234)

n_obs = 500
n_vars = 100

W_and_T = np.random.normal(size=(n_obs, n_vars))
W = W_and_T[:, 10:]
T = W_and_T[:, :10]
theta = np.array([3., 3., 3.])
y = np.dot(T[:, :3], theta) + np.random.standard_normal(size=(n_obs,))
est = LinearDML()
### Estimate with OLS confidence intervals
est.fit(y, T, W=W, inference=BootstrapInference(200, multiplier_type='normal')) # W -> high-dimensional confounders, X -> features
