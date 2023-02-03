import unittest
import pytest
import pickle
import numpy as np
from contextlib import ExitStack
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.linear_model import (LinearRegression, LassoCV, Lasso, MultiTaskLasso,
                                  MultiTaskLassoCV, LogisticRegression)
from econml.panel.dml import DynamicDML
from econml.panel.dml._dml import _get_groups_period_filter
from econml.inference import BootstrapInference, EmpiricalInferenceResults, NormalInferenceResults
from econml.utilities import shape, hstack, vstack, reshape, cross_product
import econml.tests.utilities  # bugfix for assertWarns
from econml.tests.dgp import DynamicPanelDGP

n_panels = 100  # number of panels
n_periods = 3  # number of time periods per panel
n = n_panels * n_periods
groups = np.repeat(a=np.arange(n_panels), repeats=n_periods, axis=0)

def make_random(n, is_discrete, d):
    if d is None:
        return None
    sz = (n, d) if d >= 0 else (n,)
    if is_discrete:
        return np.random.choice(['a', 'b', 'c'], size=sz)
    else:
        return np.random.normal(size=sz)

d_t = 2
is_discrete = True if d_t <= 1 else False
# for is_discrete in [False]:
d_y = 3
d_x = 2
d_w = 2

W, X, Y, T = [make_random(n, is_discrete, d)
                for is_discrete, d in [(False, d_w),
                                        (False, d_x),
                                        (False, d_y),
                                        (is_discrete, d_t)]]
T_test = np.hstack([(T.reshape(-1, 1) if d_t == -1 else T) for i in range(n_periods)])
featurizer, fit_cate_intercept = PolynomialFeatures(degree=2, include_bias=False), True

d_t_final = (2 if is_discrete else max(d_t, 1)) * n_periods

effect_shape = (n,) + ((d_y,) if d_y > 0 else ())
effect_summaryframe_shape = (n * (d_y if d_y > 0 else 1), 6)
marginal_effect_shape = ((n,) +
                            ((d_y,) if d_y > 0 else ()) +
                            ((d_t_final,) if d_t_final > 0 else ()))
marginal_effect_summaryframe_shape = (n * (d_y if d_y > 0 else 1) *
                                        (d_t_final if d_t_final > 0 else 1), 6)

# since T isn't passed to const_marginal_effect, defaults to one row if X is None
const_marginal_effect_shape = ((n if d_x else 1,) +
                                ((d_y,) if d_y > 0 else ()) +
                                ((d_t_final,) if d_t_final > 0 else ()))
const_marginal_effect_summaryframe_shape = (
    (n if d_x else 1) * (d_y if d_y > 0 else 1) *
    (d_t_final if d_t_final > 0 else 1), 6)

fd_x = featurizer.fit_transform(X).shape[1:] if featurizer and d_x\
    else ((d_x,) if d_x else (0,))
coef_shape = Y.shape[1:] + (d_t_final, ) + fd_x

coef_summaryframe_shape = (
    (d_y if d_y > 0 else 1) * (fd_x[0] if fd_x[0] >
                                0 else 1) * (d_t_final), 6)
intercept_shape = Y.shape[1:] + (d_t_final, )
intercept_summaryframe_shape = (
    (d_y if d_y > 0 else 1) * (d_t_final if d_t_final > 0 else 1), 6)

est = DynamicDML(model_y=Lasso() if d_y < 1 else MultiTaskLasso(),
                    model_t=LogisticRegression() if is_discrete else
                    (Lasso() if d_t < 1 else MultiTaskLasso()),
                    featurizer=featurizer,
                    fit_cate_intercept=fit_cate_intercept,
                    discrete_treatment=is_discrete)

# ensure we can serialize the unfit estimator
pickle.dumps(est)

# inf = None
inf = BootstrapInference(2, only_final=False)

print(Y.shape)
est.fit(Y, T, X=X, W=W, groups=groups, inference=inf)
print("******************")
inf = BootstrapInference(2, only_final=True)

est.fit(Y, T, X=X, W=W, groups=groups, inference=inf)

# ensure we can pickle the fit estimator
# pickle.dumps(est)

# make sure we can call the marginal_effect and effect methods
# const_marg_eff = est.const_marginal_effect(X)
# marg_eff = est.marginal_effect(T_test, X)
