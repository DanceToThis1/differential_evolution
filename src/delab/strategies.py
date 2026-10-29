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

def current_to_pbest_1(pop, fitness, F, p, rng, archive=None):
    """为每个目标个体生成 pop[i] + F[i] * (pop[pbest] - pop[i]) + F[i] * (pop[b] - pop[c])。

    pop 是形状为 (n, dim) 的数值数组，fitness 是长度为 n 的数组，F 是长度为 n 的数组。
    p 是一个 0 到 1 之间的浮点数，表示选择 pbest 个体的比例；调用者保证 n >= 4。
    archive 是一个可选的形状为 (m, dim) 的数组，表示归档个体；如果提供，b 始终从当前种群选择；
    c 从当前种群与归档的拼接集合中选择。返回新的浮点数组，不修改 pop、fitness、F、archive。
    rng 负责抽样，调用后其随机状态会推进；边界修复由优化器单独完成。
    """
    n = len(pop)
    idxs = np.arange(n)
    mutants = np.empty_like(pop, dtype=float)
    pbest_count = max(1, int(np.ceil(p * n)))
    pbest_idxs = np.argsort(fitness)[:pbest_count]
    for i in range(n):
        pbest_idx = rng.choice(pbest_idxs)
        if archive is not None and len(archive) > 0:
            combined_pop = np.vstack((pop, archive))
            combined_idxs = np.arange(len(combined_pop))
            b = rng.choice(idxs[idxs != i])
            candidates = combined_idxs[(combined_idxs != i) & (combined_idxs != b)]
            c = rng.choice(candidates)
            mutants[i] = (
                pop[i]
                + F[i] * (pop[pbest_idx] - pop[i])
                + F[i] * (combined_pop[b] - combined_pop[c])
            )
        else:
            b, c = rng.choice(idxs[idxs != i], size=2, replace=False)
            mutants[i] = (
                pop[i]
                + F[i] * (pop[pbest_idx] - pop[i])
                + F[i] * (pop[b] - pop[c])
            )
    return mutants