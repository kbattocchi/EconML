import numpy as np
import scipy
import time
from econml.dml import LinearDML
from econml.inference import BootstrapInference
import matplotlib.pyplot as plt

all_bootstrapping_times = []
final_bootstrapping_times = []
all_n_bootstap_samples = [1] + [40 * i for i in range(1, 15)]

X = np.random.normal(size=(1000, 5))
T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))

for n_bootstrap_samples in all_n_bootstap_samples:
    est = LinearDML(discrete_treatment=True)
    start_time = time.time()
    est.fit(y, T, X=X, W=None, inference=BootstrapInference(n_bootstrap_samples=n_bootstrap_samples, only_final=False))
    end_time = time.time()
    elapsed_time = end_time - start_time
    all_bootstrapping_time = elapsed_time
    all_bootstrapping_times.append(all_bootstrapping_time)

    est = LinearDML(discrete_treatment=True)
    start_time = time.time()
    est.fit(y, T, X=X, W=None, inference=BootstrapInference(n_bootstrap_samples=n_bootstrap_samples, only_final=True))
    end_time = time.time()
    elapsed_time = end_time - start_time
    final_bootstrapping_time = elapsed_time
    final_bootstrapping_times.append(final_bootstrapping_time)

plt.plot(all_n_bootstap_samples, all_bootstrapping_times, c="r", marker='o', label="all")
plt.plot(all_n_bootstap_samples, final_bootstrapping_times, c="b", marker='o', label="final only")
plt.xlabel("Number of Bootstrap Samples")
plt.ylabel("Runtime in Seconds")
plt.legend()
plt.title("Runtime Comparison of Bootstrapping Methods")
plt.show()