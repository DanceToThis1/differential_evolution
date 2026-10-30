from types import SimpleNamespace

import numpy as np

from delab import JDE


def sphere(x):
    return float(np.sum(x**2))


def test_jde_stores_initial_parameters_for_each_individual():
    optimizer = JDE(sphere, [(-5, 5)], popsize=4, F=0.6, CR=0.8)

    np.testing.assert_array_equal(optimizer.F_i, [0.6, 0.6, 0.6, 0.6])
    np.testing.assert_array_equal(optimizer.CR_i, [0.8, 0.8, 0.8, 0.8])
    assert optimizer.nfev == 0


def test_jde_generates_candidates_without_changing_saved_parameters(monkeypatch):
    optimizer = JDE(sphere, [(-5, 5)], popsize=4)
    # 前两次 random 决定谁重采样：F 的前两个个体、CR 的后两个个体。
    # 恰好为 0.1 时不重采样，因为判断条件是 < 0.1。
    draws = iter(
        [
            np.array([0.0, 0.09, 0.1, 0.99]),
            np.array([0.1, 0.99, 0.09, 0.0]),
            np.array([0.2, 0.8]),
        ]
    )

    def random(size):
        values = next(draws)
        assert values.shape == (size,)
        return values

    def uniform(low, high, size):
        assert (low, high, size) == (0.1, 1.0, 2)
        return np.array([0.25, 0.75])

    # 控制随机输入，让测试不依赖“这次运气是否刚好触发重采样”。
    monkeypatch.setattr(
        optimizer, "rng", SimpleNamespace(random=random, uniform=uniform)
    )
    F, CR = optimizer.generate_params()

    np.testing.assert_allclose(F, [0.25, 0.75, 0.5, 0.5])
    np.testing.assert_allclose(CR, [0.9, 0.9, 0.2, 0.8])
    np.testing.assert_array_equal(optimizer.F_i, [0.5] * 4)
    np.testing.assert_array_equal(optimizer.CR_i, [0.9] * 4)
    assert not np.shares_memory(F, optimizer.F_i)
    assert not np.shares_memory(CR, optimizer.CR_i)


def test_jde_only_keeps_parameters_of_successful_individuals():
    optimizer = JDE(sphere, [(-5, 5)], popsize=4)
    success = np.array([True, False, True, False])
    F = np.array([0.2, 0.4, 0.6, 0.8])
    CR = np.array([0.1, 0.3, 0.5, 0.7])

    optimizer.adapt(success, F, CR, np.array([1.0, 0.0, 2.0, 0.0]))

    np.testing.assert_allclose(optimizer.F_i, [0.2, 0.5, 0.6, 0.5])
    np.testing.assert_allclose(optimizer.CR_i, [0.1, 0.9, 0.5, 0.9])


def test_jde_step_updates_individual_and_parameters_together(monkeypatch):
    optimizer = JDE(sphere, [(-5, 5)], popsize=4, max_nfev=8)
    # 准备确定的父代和试验向量：依次是变好、变差、相等、变好。
    optimizer.population = np.array([[1.0], [2.0], [3.0], [4.0]])
    optimizer.fitness = optimizer.evaluate(optimizer.population)
    mutants = np.array([[0.0], [3.0], [3.0], [2.0]])
    F = np.array([0.2, 0.4, 0.6, 0.8])
    CR = np.array([0.1, 0.3, 0.5, 0.7])
    monkeypatch.setattr(optimizer, "generate_params", lambda: (F, CR))
    monkeypatch.setattr(optimizer, "mutate", lambda _: mutants)

    optimizer.step()

    np.testing.assert_array_equal(optimizer.population[:, 0], [0.0, 2.0, 3.0, 2.0])
    np.testing.assert_array_equal(optimizer.fitness, [0.0, 4.0, 9.0, 4.0])
    np.testing.assert_allclose(optimizer.F_i, [0.2, 0.5, 0.5, 0.8])
    np.testing.assert_allclose(optimizer.CR_i, [0.1, 0.9, 0.9, 0.7])
    assert optimizer.nfev == 8


def test_jde_reuses_saved_parameters_when_no_refresh_is_triggered(monkeypatch):
    optimizer = JDE(sphere, [(-5, 5)], popsize=4)
    optimizer.F_i[:] = [0.2, 0.4, 0.6, 0.8]
    optimizer.CR_i[:] = [0.1, 0.3, 0.5, 0.7]
    # 所有重采样判断都为 False，这代应该完整继承已保存的参数。
    monkeypatch.setattr(
        optimizer,
        "rng",
        SimpleNamespace(
            random=lambda size: np.full(size, 0.5),
            uniform=lambda low, high, size: np.zeros(size),
        ),
    )

    F, CR = optimizer.generate_params()

    np.testing.assert_array_equal(F, [0.2, 0.4, 0.6, 0.8])
    np.testing.assert_array_equal(CR, [0.1, 0.3, 0.5, 0.7])


def test_jde_converges_and_reproduces_results_with_same_seed():
    optimizers = [
        JDE(sphere, [(-5, 5)] * 2, popsize=30, max_nfev=5000, seed=42) for _ in range(2)
    ]
    first, second = [optimizer.run() for optimizer in optimizers]
    assert first is not None
    assert second is not None
    assert first.best_f < 1e-6
    np.testing.assert_allclose(first.best_f, sphere(first.best_x), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(first.best_x, second.best_x)
    np.testing.assert_array_equal(optimizers[0].F_i, optimizers[1].F_i)
    np.testing.assert_array_equal(optimizers[0].CR_i, optimizers[1].CR_i)
    assert first.best_f == second.best_f
    assert first.nfev == second.nfev == 4980
    assert first.generation == second.generation
