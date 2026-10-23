from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass
class GenerationInfo:
    """每一代结束后产出的信息，用于画图和分析。"""
    generation: int
    nfev: int              # 到目前为止的函数评估次数
    best_x: np.ndarray
    best_f: float
    extra: dict            # 子类想额外记录的东西，比如 JADE 的 mean_F、mean_CR

class Optimizer(ABC):

    def __init__(self, func, bounds, popsize=50, max_nfev=100_000, seed=None):
        self.func = func
        self.bounds = np.array(bounds, dtype=float, copy=True)
        self.popsize = popsize
        self.max_nfev = max_nfev
        self.dim = len(self.bounds)
        self.nfev = 0
        self.rng = np.random.default_rng(seed)

    def iterate(self):
       """生成器：初始化后，每完成一代就 yield 一个 GenerationInfo。"""
       if self.nfev > 0:
           raise RuntimeError("iterate() should be called before any evaluations.")
       if self.max_nfev < self.popsize:
           raise ValueError("max_nfev must be at least popsize.")
       lower, upper = self.bounds[:, 0], self.bounds[:, 1]
       self.population = self.rng.uniform(lower, upper, size=(self.popsize, self.dim))
       self.fitness = self.evaluate(self.population)
       generation = 0
       while True:
           best_idx = np.argmin(self.fitness)
           yield GenerationInfo(generation=generation, nfev=self.nfev, best_x=self.population[best_idx].copy(), best_f=self.fitness[best_idx], extra=self.extra_info())
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
        self.population[success] = trials[success]
        self.fitness[success] = trial_fitness[success]
        self.adapt(success, F, CR, improvement)


    def evaluate(self, X):
        if self.nfev + len(X) > self.max_nfev:
            raise RuntimeError("Maximum number of function evaluations exceeded.")
        self.nfev += len(X)
        return np.array([self.func(x) for x in X], dtype=float)

    def repair(self, mutants, targets):
        """越界处理，默认用 JADE 里的“和边界取中点”。"""
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        repaired = np.where(mutants < lower, (mutants + targets) / 2, mutants)
        repaired = np.where(repaired > upper, (repaired + targets) / 2, repaired)
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