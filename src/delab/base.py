from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass
class GenerationInfo:
    """每一代结束后产出的信息，用于画图和分析。"""

    generation: int
    nfev: int  # 到目前为止的函数评估次数
    best_x: np.ndarray
    best_f: float
    extra: dict  # 子类想额外记录的东西，比如 JADE 的 mean_F、mean_CR


class Optimizer(ABC):
    def __init__(self, func, bounds, popsize=50, max_nfev=100_000, seed=None):
        if not callable(func):
            raise TypeError("func must be callable.")

        # 个数必须是整数；bool 虽然是 int 的子类，但不应该作为个数。
        for name, value in (("popsize", popsize), ("max_nfev", max_nfev)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if max_nfev < popsize:
            raise ValueError("max_nfev must be at least popsize.")

        try:
            bounds = np.array(bounds, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "bounds must contain numeric (lower, upper) pairs."
            ) from exc
        if bounds.ndim != 2 or bounds.shape[0] == 0 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dim, 2), with dim >= 1.")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("bounds must contain only finite values.")
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError(
                "Each lower bound must be less than or equal to its upper bound."
            )

        self.func = func
        self.bounds = bounds
        self.popsize = int(popsize)
        self.max_nfev = int(max_nfev)
        self.dim = len(self.bounds)
        self.nfev = 0
        self.rng = np.random.default_rng(seed)

    def iterate(self):
        """生成器：初始化后，每完成一代就 yield 一个 GenerationInfo。"""
        if self.nfev > 0:
            raise RuntimeError("iterate() should be called before any evaluations.")
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        self.population = self.rng.uniform(lower, upper, size=(self.popsize, self.dim))
        self.fitness = self.evaluate(self.population)
        generation = 0
        while True:
            best_idx = np.argmin(self.fitness)
            yield GenerationInfo(
                generation=generation,
                nfev=self.nfev,
                best_x=self.population[best_idx].copy(),
                best_f=self.fitness[best_idx],
                extra=self.extra_info(),
            )
            if self.nfev + self.popsize > self.max_nfev:
                break
            self.step()
            generation += 1

    def run(self):
        """跑到停止条件为止，返回最终结果。内部调用 iterate()。"""
        last_info = None
        for info in self.iterate():
            last_info = info
        return last_info

    def step(self):
        """完成一代：生成参数 → 变异 → 修复越界 → 交叉 → 评估 → 选择 → 自适应。"""
        if self.nfev + self.popsize > self.max_nfev:
            raise ValueError("Maximum number of function evaluations exceeded.")
        F, CR = self.generate_params()
        mutants = self.mutate(F)
        mutants = self.repair(mutants, self.population)
        trials = self.crossover(self.population, mutants, CR)
        trial_fitness = self.evaluate(trials)
        success = trial_fitness < self.fitness
        improvement = np.where(success, self.fitness - trial_fitness, 0.0)
        self.before_replace(success)
        self.population[success] = trials[success]
        self.fitness[success] = trial_fitness[success]
        self.adapt(success, F, CR, improvement)

    def evaluate(self, X):
        if self.nfev + len(X) > self.max_nfev:
            raise ValueError("Maximum number of function evaluations exceeded.")
        self.nfev += len(X)
        return np.array([self.func(x) for x in X], dtype=float)

    def repair(self, mutants, targets):
        """越界处理，默认用 JADE 里的“和边界取中点”。"""
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        repaired = np.where(mutants < lower, (targets + lower) / 2, mutants)
        repaired = np.where(mutants > upper, (targets + upper) / 2, repaired)
        return repaired

    def crossover(self, targets, mutants, CR):
        """二项交叉，保证至少有一维来自变异向量。"""
        n = len(targets)
        mask = self.rng.random(targets.shape) < CR[:, None]
        forced_dims = self.rng.integers(self.dim, size=n)
        mask[np.arange(n), forced_dims] = True

        trials = np.where(mask, mutants, targets)
        return trials

    # ---------- 子类必须 / 可以重写的钩子 ----------
    @abstractmethod
    def generate_params(self):
        """返回 F 和 CR，各是长度为 popsize 的数组。"""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def mutate(self, F):
        """返回变异向量矩阵，形状 (popsize, dim)。"""
        raise NotImplementedError("Subclasses should implement this method.")

    def adapt(self, success, F, CR, improvement):
        """每代结束调用。success 是布尔数组，improvement 是适应值改进量。
        默认什么都不做（基础 DE 就不需要）。"""
        pass

    def extra_info(self):
        """返回想记录到 GenerationInfo.extra 的数据，默认返回空字典。"""
        return {}

    def before_replace(self, success):
        """在替换父代之前调用"""
        pass

class Archive:
    """归档个体的简单实现。"""
    def __init__(self, max_size, rng):
        self.max_size = max_size
        self.members = np.empty((0, 0), dtype=float)
        self.rng = rng

    def add(self, vectors):
        """添加新成员，可能会丢弃旧成员以保持最大容量。"""
        if self.members.size == 0:
            self.members = vectors.copy()
        else:
            self.members = np.vstack((self.members, vectors))

    def trim(self):
        """如果归档成员数超过最大容量，则随机丢弃一些成员。"""
        if len(self.members) > self.max_size:
            indices_to_keep = self.rng.choice(
                len(self.members), size=self.max_size, replace=False
            )
            self.members = self.members[indices_to_keep]

    def as_array(self):
        """返回归档成员的数组视图。"""
        return self.members