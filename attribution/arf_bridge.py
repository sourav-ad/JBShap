import os
import numpy as np
import pandas as pd

from rpy2.robjects import r, default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


class ARFDensity:
    def __init__(self,
    num_trees=30,
    n_jobs=4,
    seed=10,
    low_density_threshold=None,
    low_density_shrink=1.0):
        self.num_trees = num_trees
        self.n_jobs = n_jobs
        self.seed = seed

        self.low_density_threshold = low_density_threshold
        self.low_density_shrink = low_density_shrink

        self.arf = importr("arf")
        importr("doParallel")

        r(f"doParallel::registerDoParallel(cores={n_jobs})")

        self.model = None
        self.params = None
        self.columns = None
        self.log_density_max = None

    def _to_r(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        df = pd.DataFrame(X, columns=self.columns)

        with localconverter(default_converter + pandas2ri.converter):
            return pandas2ri.py2rpy(df)

    def fit(self, X):
        X = np.asarray(X)

        self.columns = [f"x{i}" for i in range(X.shape[1])]
        r_train = self._to_r(X)

        r(f"set.seed({self.seed}, kind=\"L'Ecuyer-CMRG\")")

        self.model = self.arf.adversarial_rf(
            r_train,
            num_trees=self.num_trees,
            parallel=True,
        )

        self.params = self.arf.forde(
            self.model,
            r_train,
            family="truncnorm",
            oob=False,
            parallel=True,
        )

        train_ll = self.log_density(X)
        self.log_density_max = np.max(train_ll)

        print(
            f"[ARF] fitted | trees={self.num_trees} | "
            f"workers={self.n_jobs} | "
            f"loglik=[{train_ll.min():.3f}, {train_ll.max():.3f}]"
        )

        return self

    def log_density(self, X):
        r_x = self._to_r(X)

        ll = self.arf.lik(
            self.params,
            r_x,
            arf=self.model,
            oob=False,
            log=True,
            parallel=True,
        )

        return np.asarray(ll, dtype=float).flatten()

    def score(self, X):
        ll = self.log_density(X)

        # Relative ARF density.
        # Training-data maximum has score 1, but query points are
        # allowed to exceed 1. Do NOT clip.
        score = np.exp(ll - self.log_density_max)

        # Optional diagnostic: more aggressively suppress
        # regions ARF itself regards as low-density.
        if self.low_density_threshold is not None:
            low_mask = score < self.low_density_threshold
            score = score.copy()
            score[low_mask] *= self.low_density_shrink

        return score

    def density_diagnostics(self, X, threshold=0.1):
        ll = self.log_density(X)
        score = np.exp(ll - self.log_density_max)

        return {
            "min": float(np.min(score)),
            "median": float(np.median(score)),
            "max": float(np.max(score)),
            "frac_below_threshold": float(np.mean(score < threshold)),
            "threshold": float(threshold),
        }