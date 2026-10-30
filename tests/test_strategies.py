from types import SimpleNamespace

import numpy as np
import pytest

from delab.strategies import current_to_pbest_1, rand_1


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


def test_current_to_pbest_without_archive_matches_hand_calculation():
    pop = np.array([[4.0], [0.0], [8.0], [2.0]])
    fitness = np.array([16.0, 0.0, 64.0, 4.0])
    F = np.array([0.5, 0.25, 1.0, 0.5])
    # p=0.5，前两名的索引是 [1, 3]。
    # 每个个体依次抽 pbest、[b, c]；每行是（候选索引，固定返回值）。
    draws = iter(
        [
            ([1, 3], 3),
            ([1, 2, 3], [1, 2]),  # i=0
            ([1, 3], 1),
            ([0, 2, 3], [2, 3]),  # i=1
            ([1, 3], 3),
            ([0, 1, 3], [3, 0]),  # i=2
            ([1, 3], 1),
            ([0, 1, 2], [0, 1]),  # i=3
        ]
    )

    def choice(candidates, size=None, replace=True):
        expected_candidates, selected = next(draws)
        np.testing.assert_array_equal(candidates, expected_candidates)
        if size is not None:
            assert size == 2
            assert replace is False
        return selected

    rng = SimpleNamespace(choice=choice)
    mutants = current_to_pbest_1(pop, fitness, F, p=0.5, rng=rng)

    # 第 0 行：4 + 0.5 * (2 - 4) + 0.5 * (0 - 8) = -1。
    expected = np.array([[-1.0], [1.5], [0.0], [3.0]])
    np.testing.assert_allclose(mutants, expected, rtol=0, atol=1e-12)


def test_current_to_pbest_with_archive_matches_hand_calculation():
    pop = np.array([[4.0], [0.0], [8.0], [2.0]])
    fitness = np.array([16.0, 0.0, 64.0, 4.0])
    F = np.array([0.5, 0.25, 1.0, 0.5])
    archive = np.array([[-2.0], [10.0]])
    # 拼接后索引 4、5 对应归档；每个个体依次抽 pbest、b、c。
    # b 的候选只含种群；c 的候选排除了 i、b，并包含归档。
    draws = iter(
        [
            ([1, 3], 3),
            ([1, 2, 3], 1),
            ([2, 3, 4, 5], 4),  # i=0，c 来自归档
            ([1, 3], 1),
            ([0, 2, 3], 2),
            ([0, 3, 4, 5], 0),  # i=1，c 来自种群
            ([1, 3], 3),
            ([0, 1, 3], 3),
            ([0, 1, 4, 5], 5),  # i=2，c 来自归档
            ([1, 3], 1),
            ([0, 1, 2], 0),
            ([1, 2, 4, 5], 1),  # i=3，c 来自种群
        ]
    )

    def choice(candidates):
        expected_candidates, selected = next(draws)
        np.testing.assert_array_equal(candidates, expected_candidates)
        return selected

    rng = SimpleNamespace(choice=choice)
    mutants = current_to_pbest_1(pop, fitness, F, p=0.5, rng=rng, archive=archive)

    # 第 0 行的 c=4 对应 -2：4 + 0.5 * (2 - 4) + 0.5 * (0 - (-2)) = 4。
    expected = np.array([[4.0], [1.0], [-6.0], [3.0]])
    np.testing.assert_allclose(mutants, expected, rtol=0, atol=1e-12)
