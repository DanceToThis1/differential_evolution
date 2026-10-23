import numpy as np
from .base import Optimizer

class DE(Optimizer):
    def __init__(self, func, bounds, popsize=50, max_nfev=100_000, seed=None, F=0.5, CR=0.9):
        super().__init__(func=func, bounds=bounds, popsize=popsize, max_nfev=max_nfev, seed=seed)
        self.F = F
        self.CR = CR

    def generate_params(self):
        """返回 F 和 CR，各是长度为 popsize 的数组。"""
        return np.full(self.popsize, self.F), np.full(self.popsize, self.CR)

    def mutate(self, F):
        if self.popsize < 4:
            raise ValueError("Population size must be at least 4 for mutation.")
        """返回变异向量矩阵，形状 (popsize, dim)。"""
        idxs = np.arange(self.popsize)
        mutants = np.empty_like(self.population)
        for i in range(self.popsize):
            a, b, c = self.rng.choice(idxs[idxs != i], size=3, replace=False)
            mutants[i] = self.population[a] + F[i] * (self.population[b] - self.population[c])
        return mutants