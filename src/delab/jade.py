import numpy as np
from .base import Optimizer
from .archive import Archive
from .strategies import current_to_pbest_1

class JADE(Optimizer):
    def __init__(
        self,
        func,
        bounds,
        popsize=50,
        max_nfev=100_000,
        seed=None,
        p=0.1,
        c=0.1,
    ):
        super().__init__(
            func=func, bounds=bounds, popsize=popsize, max_nfev=max_nfev, seed=seed
        )
        if self.popsize < 3:
            raise ValueError("popsize must be at least 3 for JADE.")
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1].")
        if not 0 < c <= 1:
            raise ValueError("c must be in (0, 1].")
        self.p = float(p)
        self.c = float(c)
        self.archive = Archive(max_size=popsize, rng=self.rng)
        self.mean_F = 0.5
        self.mean_CR = 0.5

    def generate_params(self):
        """返回 F 和 CR，各是长度为 popsize 的数组。"""
        mean_F = self.mean_F
        mean_CR = self.mean_CR

        F = self.rng.standard_cauchy(size=self.popsize) * 0.1 + mean_F
        invalid = F <= 0
        while np.any(invalid):
            F[invalid] = self.rng.standard_cauchy(size=np.count_nonzero(invalid)) * 0.1 + mean_F
            invalid = F <= 0
        F = np.clip(F, 0, 1)
        CR = np.clip(
            self.rng.normal(loc=mean_CR, scale=0.1, size=self.popsize), 0, 1
        )
        return F, CR

    def mutate(self, F):
        """返回变异向量矩阵，形状 (popsize, dim)。"""
        return current_to_pbest_1(
            self.population,
            self.fitness,
            F,
            self.p,
            self.rng,
            archive=self.archive.as_array(),
        )

    def adapt(self, success, F, CR, improvement):
        """每代结束调用。success 是布尔数组，improvement 是适应值改进量。"""
        successful_F = F[success]
        successful_CR = CR[success]
        if len(successful_F) > 0:
            self.mean_CR = (1 - self.c) * self.mean_CR + self.c * np.mean(successful_CR)
            lehmer_mean_F = np.sum(successful_F**2) / np.sum(successful_F)
            self.mean_F = (1 - self.c) * self.mean_F + self.c * lehmer_mean_F

    def before_replace(self, success):
        """在替换父代之前调用"""
        self.archive.add(self.population[success])
        self.archive.trim()
