# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bootstrap sampling."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from scipy.stats import norm
from collections import OrderedDict
import pandas as pd

from ..utilities import convertArg

class MultiplierBootstrapEstimator:

    def __init__(self, wrapped,
                 n_bootstrap_samples=100,
                 n_jobs=None,
                 verbose=0,
                 compute_means=True,
                 multiplier_type='Bayes',
                 bootstrap_type='pivot'):
        self._instances = [clone(wrapped, safe=False) for _ in range(n_bootstrap_samples)]
        self._n_bootstrap_samples = n_bootstrap_samples
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._multiplier_type = multiplier_type
        self._compute_means = compute_means
        self._bootstrap_type = bootstrap_type
        self._wrapped = wrapped
        self._multiplier_weights = {}
        self._length_coef = None
        self._length_intercept = None

    def _compute_multiplier_weights(self, model_type):
        # Adapted from Double DML
        n_samples = self._wrapped.model_final_._n_samples
        if model_type == "coef":
            self._length_coef = np.prod(self._wrapped.coef_.shape)
        elif model_type == "intercept":
            self._length_coef = np.prod(self._wrapped.intercept_.shape)
        else:
            raise ValueError("Unsupported model type for multiplier bootstrap")
        if self._multiplier_type == 'Bayes':
            weights = np.random.exponential(scale=1.0, size=(self._n_bootstrap_samples, n_samples)) - 1.
        elif self._multiplier_type == 'normal':
            weights = np.random.normal(loc=0.0, scale=1.0, size=(self._n_bootstrap_samples, n_samples))
        elif self._multiplier_type == 'wild':
            xx = np.random.normal(loc=0.0, scale=1.0, size=(self._n_bootstrap_samples, n_samples))
            yy = np.random.normal(loc=0.0, scale=1.0, size=(self._n_bootstrap_samples, n_samples))
            weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1) / 2
        else:
            raise ValueError('invalid multiplier bootstrap method')
        self._multiplier_weights[model_type] = weights
        self._current_multiplier_weight = self._multiplier_weights[model_type][0]

    def fit(self, *args, **named_args):
        """
        Fit the model.

        The full signature of this method is the same as that of the wrapped object's `fit` method.
        """
        
        # In the fit, we want to multiply the coefficient by various multiplier samples.
        self._wrapped._set_bootstrap_params("multiplier", None, self._n_jobs, self._verbose)
        self._wrapped.fit(*args, **named_args)
        self._instances = [clone(self._wrapped, safe=False)]
        self._compute_multiplier_weights("coef")
        self._compute_multiplier_weights("intercept")
        return self

    def __getattr__(self, name):
        """
        Get proxy attribute that wraps the corresponding attribute with the same name from the wrapped object.

        Additionally, the suffix "_interval" is supported for getting an interval instead of a point estimate.
        """
        
        # don't proxy special methods
        if name.startswith('__'):
            raise AttributeError(name)

        def proxy(make_call, name, summary):
            def summarize_with(f):
                instance_results = []
                obj = clone(self._wrapped, safe=False)
                for i in range(self._n_bootstrap_samples):
                    if 'coef' in name:
                        self._current_multiplier_weight = self._multiplier_weights["coef"][i]
                    elif 'intercept' in name:
                        self._current_multiplier_weight = self._multiplier_weights["intercept"][i]
                    instance_results.append(f(obj, name))
                instance_results = np.array(instance_results)
                results = instance_results, f(self._wrapped, name)
                return summary(*results)
            if make_call:
                def call(*args, **kwargs):
                    return summarize_with(lambda obj, name: getattr(obj, name)(*args, **kwargs))
                return call
            else:
                return summarize_with(lambda obj, name: getattr(obj, name))

        def get_mean():
            # for attributes that exist on the wrapped object, just compute the mean of the wrapped calls
            return proxy(callable(getattr(self._wrapped, name)), name, lambda arr, _: np.mean(arr, axis=0))

        def get_std():
            prefix = name[: - len('_std')]
            return proxy(callable(getattr(self._wrapped, prefix)), prefix,
                         lambda arr, _: np.std(arr, axis=0))

        def get_interval():
            # if the attribute exists on the wrapped object once we remove the suffix,
            # then we should be computing a confidence interval for the wrapped calls
            prefix = name[: - len("_interval")]

            def call_with_bounds(can_call, lower, upper):
                if ('coef' not in prefix) and ('intercept' not in prefix):
                    def percentile_bootstrap(arr, _):
                        return np.percentile(arr, lower, axis=0), np.percentile(arr, upper, axis=0)

                    def pivot_bootstrap(arr, est):
                        return 2 * est - np.percentile(arr, upper, axis=0), 2 * est - np.percentile(arr, lower, axis=0)

                    def normal_bootstrap(arr, est):
                        std = np.std(arr, axis=0)
                        return est - norm.ppf(upper / 100) * std, est - norm.ppf(lower / 100) * std

                    # TODO: studentized bootstrap? this would be more accurate in most cases but can we avoid
                    #       second level bootstrap which would be prohibitive computationally?

                    fn = {'percentile': percentile_bootstrap,
                        'normal': normal_bootstrap,
                        'pivot': pivot_bootstrap}[self._bootstrap_type]
                else:
                    def all_multiplier_bootstrap(arr, est):
                        var = self._wrapped.model_final_._var
                        N = self._wrapped.model_final_._n_samples
                        std_err = np.sqrt(np.diag(var)) / np.sqrt(N)
                        def multiplier_bootstrap(arr, est, i):
                            if 'coef' in name:
                                self._current_multiplier_weight = self._multiplier_weights["coef"][i]
                            elif 'intercept' in name:
                                self._current_multiplier_weight = self._multiplier_weights["intercept"][i]
                            est_shape = np.shape(est)
                            if est_shape[-1] == 0:
                                return np.empty(est_shape), np.empty(est_shape)
                            boot_t_stat = self._wrapped.model_final_._compute_mult_boot_t_stat(self._current_multiplier_weight)
                            return np.amax(np.abs(boot_t_stat))
                        all_boot_t_stat = np.zeros(self._n_bootstrap_samples)
                        for i in range(self._n_bootstrap_samples):
                            all_boot_t_stat[i] = multiplier_bootstrap(arr, est, i)
                        hatc = np.quantile(all_boot_t_stat, upper / 100)
                        return est - hatc * std_err, est + hatc * std_err
                    fn = all_multiplier_bootstrap

                return proxy(can_call, prefix, fn)

            can_call = callable(getattr(self._wrapped, prefix))
            if can_call:
                # collect extra arguments and pass them through, if the wrapped attribute was callable
                def call(*args, lower=5, upper=95, **kwargs):
                    return call_with_bounds(can_call, lower, upper)(*args, **kwargs)
                return call
            else:
                # don't pass extra arguments if the wrapped attribute wasn't callable to begin with
                def call(lower=5, upper=95):
                    return call_with_bounds(can_call, lower, upper)
                return call

        def get_inference():
            # can't import from econml.inference at top level without creating cyclical dependencies
            from ._inference import EmpiricalInferenceResults, NormalInferenceResults
            from .._cate_estimator import LinearModelFinalCateEstimatorDiscreteMixin

            prefix = name[: - len("_inference")]

            def fname_transformer(x):
                return x

            if prefix in ['const_marginal_effect', 'marginal_effect', 'effect']:
                inf_type = 'effect'
            elif prefix == 'coef_':
                inf_type = 'coefficient'
                if (hasattr(self._wrapped, 'cate_feature_names') and
                        callable(self._wrapped.cate_feature_names)):
                    def fname_transformer(x):
                        return self._wrapped.cate_feature_names(x)
            elif prefix == 'intercept_':
                inf_type = 'intercept'
            else:
                raise AttributeError("Unsupported inference: " + name)

            d_t = self._wrapped._d_t[0] if self._wrapped._d_t else 1
            if prefix == 'effect' or (isinstance(self._wrapped, LinearModelFinalCateEstimatorDiscreteMixin) and
                                      (inf_type == 'coefficient' or inf_type == 'intercept')):
                d_t = None
            d_y = self._wrapped._d_y[0] if self._wrapped._d_y else 1

            can_call = callable(getattr(self._wrapped, prefix))

            kind = self._bootstrap_type
            if kind == 'percentile' or kind == 'pivot':
                def get_dist(est, arr):
                    if kind == 'percentile':
                        return arr
                    elif kind == 'pivot':
                        return 2 * est - arr
                    else:
                        raise ValueError("Invalid kind, must be either 'percentile' or 'pivot'")

                def get_result():
                    return proxy(can_call, prefix,
                                 lambda arr, est: EmpiricalInferenceResults(
                                     d_t=d_t, d_y=d_y,
                                     pred=est, pred_dist=get_dist(est, arr),
                                     inf_type=inf_type,
                                     fname_transformer=fname_transformer,
                                     feature_names=self._wrapped.cate_feature_names(),
                                     output_names=self._wrapped.cate_output_names(),
                                     treatment_names=self._wrapped.cate_treatment_names()
                                 ))

                # Note that inference results are always methods even if the inference is for a property
                # (e.g. coef__inference() is a method but coef_ is a property)
                # Therefore we must insert a lambda if getting inference for a non-callable
                return get_result() if can_call else get_result

            else:
                assert kind == 'normal'

                def normal_inference(*args, **kwargs):
                    pred = getattr(self._wrapped, prefix)
                    if can_call:
                        pred = pred(*args, **kwargs)
                    stderr = getattr(self, prefix + '_std')
                    if can_call:
                        stderr = stderr(*args, **kwargs)
                    return NormalInferenceResults(
                        d_t=d_t, d_y=d_y, pred=pred,
                        pred_stderr=stderr, mean_pred_stderr=None, inf_type=inf_type,
                        fname_transformer=fname_transformer,
                        feature_names=self._wrapped.cate_feature_names(),
                        output_names=self._wrapped.cate_output_names(),
                        treatment_names=self._wrapped.cate_treatment_names())

                # If inference is for a property, create a fresh lambda to avoid passing args through
                return normal_inference if can_call else lambda: normal_inference()

        caught = None
        m = None
        if name.endswith("_interval"):
            m = get_interval
        elif name.endswith("_std"):
            m = get_std
        elif name.endswith("_inference"):
            m = get_inference

        # try to get interval/std first if appropriate,
        # since we don't prefer a wrapped method with this name
        if m is not None:
            try:
                return m()
            except AttributeError as err:
                caught = err
        if self._compute_means:
            return get_mean()

        raise (caught if caught else AttributeError(name))
        
