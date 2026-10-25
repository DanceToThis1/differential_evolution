"""带个体参数记忆的 jDE，复用 DE/rand/1/bin 的搜索流程。"""

import numpy as np

from .de import DE


class JDE(DE):
    """每个个体保存 F、CR，严格改进后才保留本次候选参数。"""

    def __init__(
        self, func, bounds, popsize=50, max_nfev=100_000, seed=None, F=0.5, CR=0.9
    ):
        super().__init__(
            func=func,
            bounds=bounds,
            popsize=popsize,
            max_nfev=max_nfev,
            seed=seed,
            F=F,
            CR=CR,
        )
        self.F_i = np.full(self.popsize, self.F)
        self.CR_i = np.full(self.popsize, self.CR)

    def generate_params(self):
        """分别以 10% 的概率为每个个体重新采样候选 F、CR。"""
        # 先复制：候选参数尚未通过选择，不能提前改变已保存的参数。
        F = self.F_i.copy()
        CR = self.CR_i.copy()
        refresh_F = self.rng.random(self.popsize) < 0.1
        refresh_CR = self.rng.random(self.popsize) < 0.1
        F[refresh_F] = self.rng.uniform(0.1, 1.0, size=np.count_nonzero(refresh_F))
        CR[refresh_CR] = self.rng.random(np.count_nonzero(refresh_CR))
        return F, CR

    def adapt(self, success, F, CR, improvement):
        """个体替换成功后，它使用的参数也一起保留下来。"""
        self.F_i[success] = F[success]
        self.CR_i[success] = CR[success]
