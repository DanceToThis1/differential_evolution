"""可复用的变异策略：显式接收数据，不依赖优化器实例。"""

import numpy as np


def rand_1(pop, F, rng):
    """为每个目标个体生成 pop[a] + F[i] * (pop[b] - pop[c])。

    pop 是形状为 (n, dim) 的数值数组，F 是长度为 n 的数组；调用者保证 n >= 4。
    a、b、c 互不相同，且均不等于目标索引 i。返回新的浮点数组，不修改 pop、F。
    rng 负责抽样，调用后其随机状态会推进；边界修复由优化器单独完成。
    """
    n = len(pop)
    idxs = np.arange(n)
    mutants = np.empty_like(pop, dtype=float)
    for i in range(n):
        a, b, c = rng.choice(idxs[idxs != i], size=3, replace=False)
        mutants[i] = pop[a] + F[i] * (pop[b] - pop[c])
    return mutants
