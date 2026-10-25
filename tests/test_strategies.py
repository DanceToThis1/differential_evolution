from types import SimpleNamespace

import numpy as np
import pytest

from delab.strategies import rand_1


@pytest.mark.parametrize("dtype", [int, float])
def test_rand_1_uses_distinct_other_individuals_and_each_individuals_F(dtype):
    pop = np.array([[0, 0], [1, 3], [4, 8], [8, 17]], dtype=dtype)
    F = np.array([0.5, 0.25, 1.0, 2.0])
    # 输入只读，策略若尝试原地修改种群或参数，这个测试就会报错。
    pop.setflags(write=False)
    F.setflags(write=False)
    donors = iter(enumerate(([1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2])))

    def choice(candidates, size, replace):
        target, selected = next(donors)
        assert target not in candidates
        assert len(candidates) == 3
        assert size == 3
        assert replace is False
        assert all(index in candidates for index in selected)
        return selected

    # 固定抽样结果，手算变异值；例如首行是 [1, 3] + 0.5 * ([-4, -9])。
    mutants = rand_1(pop, F, SimpleNamespace(choice=choice))

    expected = np.array([[-1, -1.5], [6, 12.25], [7, 14], [-6, -10]])
    np.testing.assert_allclose(mutants, expected, rtol=0, atol=1e-12)
    assert not np.shares_memory(mutants, pop)
