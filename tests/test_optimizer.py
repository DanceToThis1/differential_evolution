import numpy as np

from delab.de import DE


def sphere(x):
    return sum(value**2 for value in x)


def test_de_converges_on_sphere():
    optimizer = DE(
        func=sphere, bounds=[(-5, 5), (-5, 5)], popsize=30, max_nfev=5000, seed=42
    )
    result = optimizer.run()
    assert result is not None
    assert 0 <= sphere(result.best_x) < 1e-6
    np.testing.assert_allclose(result.best_f, sphere(result.best_x), rtol=0, atol=1e-12)
    assert optimizer.nfev <= 5000


def test_de_reproduces_results_with_same_seed():
    results = []
    for _ in range(2):
        optimizer = DE(
            func=sphere, bounds=[(-5, 5), (-5, 5)], popsize=30, max_nfev=5000, seed=42
        )
        results.append(optimizer.run())
    first, second = results
    np.testing.assert_array_equal(first.best_x, second.best_x)
    assert first.best_f == second.best_f
    assert first.nfev == second.nfev
    assert first.generation == second.generation


def test_repair_uses_midpoint_of_target_and_violated_bound():
    # 准备：两维使用不同边界，覆盖上下越界、合法值和边界值。
    optimizer = DE(func=sphere, bounds=[(-5, 5), (0, 10)], popsize=4)
    targets = np.array([[4.0, 2.0], [-4.0, 8.0], [1.0, 3.0], [-2.0, 7.0]])
    mutants = np.array([[20.0, -6.0], [-20.0, 16.0], [2.0, 4.0], [-5.0, 10.0]])

    # 执行：直接测试 repair，不需要先运行整个优化器。
    repaired = optimizer.repair(mutants, targets)

    # 验证：手算期望值，例如第一行是 (4 + 5) / 2 和 (2 + 0) / 2。
    expected = np.array([[4.5, 1.0], [-4.5, 9.0], [2.0, 4.0], [-5.0, 10.0]])
    np.testing.assert_allclose(repaired, expected, rtol=0, atol=1e-12)


def test_de_only_evaluates_points_within_bounds():
    def bounded_objective(x):
        # 每一次评估都检查，避免越界试验向量被淘汰后掩盖问题。
        assert np.all(x >= -5), f"Point below lower bound: {x}"
        assert np.all(x <= 5), f"Point above upper bound: {x}"
        return -float(np.sum(x))

    optimizer = DE(
        func=bounded_objective,
        bounds=[(-5, 5), (-5, 5)],
        popsize=10,
        max_nfev=100,
        seed=0,
        F=1.5,
        CR=1.0,
    )
    optimizer.run()
    assert optimizer.nfev == 100
