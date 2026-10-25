from numbers import Real

import numpy as np

from .base import Optimizer


class DE(Optimizer):
    def __init__(
        self, func, bounds, popsize=50, max_nfev=100_000, seed=None, F=0.5, CR=0.9
    ):
        super().__init__(
            func=func, bounds=bounds, popsize=popsize, max_nfev=max_nfev, seed=seed
        )
        # rand/1 需要从目标个体之外选出三个不同个体，这是 DE 的约束。
        if self.popsize < 4:
            raise ValueError("popsize must be at least 4 for DE/rand/1.")
        for name, value in (("F", F), ("CR", CR)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if F <= 0:
            raise ValueError("F must be positive.")
        if not 0 <= CR <= 1:
            raise ValueError("CR must be between 0 and 1, inclusive.")
        self.F = float(F)
        self.CR = float(CR)

    def generate_params(self):
        """返回 F 和 CR，各是长度为 popsize 的数组。"""
        return np.full(self.popsize, self.F), np.full(self.popsize, self.CR)

    def mutate(self, F):
        """返回变异向量矩阵，形状 (popsize, dim)。"""
        idxs = np.arange(self.popsize)
        mutants = np.empty_like(self.population)
        for i in range(self.popsize):
            a, b, c = self.rng.choice(idxs[idxs != i], size=3, replace=False)
            mutants[i] = self.population[a] + F[i] * (
                self.population[b] - self.population[c]
            )
        return mutants
