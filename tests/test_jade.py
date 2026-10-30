import numpy as np
import pytest

from delab import JADE


def sphere(x):
    return float(sum(x**2))


def test_jade_converges_on_sphere():
    optimizer = JADE(
        func=sphere, bounds=[(-5, 5), (-5, 5)], popsize=30, max_nfev=5000, seed=42
    )
    result = optimizer.run()
    assert result is not None
    assert 0 <= sphere(result.best_x) < 1e-6
    assert optimizer.nfev <= 5000


def test_jade_adapt_learns_from_successful_parameters():
    optimizer = JADE(sphere, [(-5, 5)], popsize=3, c=0.1)
    success = np.array([True, False, True])
    F = np.array([0.2, 1.0, 0.8])
    CR = np.array([0.2, 1.0, 0.6])

    optimizer.adapt(success, F, CR, np.array([1.0, 0.0, 1.0]))

    # 只使用成功的 F=[0.2, 0.8]，中间失败个体的 1.0 不参与计算。
    # Lehmer 平均 = (0.04 + 0.64) / (0.2 + 0.8) = 0.68。
    # 新 mean_F = 0.9 * 0.5 + 0.1 * 0.68 = 0.518。
    assert abs(optimizer.mean_F - 0.518) < 1e-12
    # 成功 CR 的平均值为 0.4，新 mean_CR = 0.9 * 0.5 + 0.1 * 0.4。
    assert abs(optimizer.mean_CR - 0.49) < 1e-12


def test_jade_adapt_keeps_centers_when_no_individual_succeeds():
    optimizer = JADE(sphere, [(-5, 5)], popsize=3)
    optimizer.mean_F = 0.7
    optimizer.mean_CR = 0.3
    success = np.array([False, False, False])
    F = np.array([0.2, 0.4, 0.8])
    CR = np.array([0.1, 0.5, 0.9])

    optimizer.adapt(success, F, CR, np.zeros(3))

    assert optimizer.mean_F == 0.7
    assert optimizer.mean_CR == 0.3


def test_jade_rejects_population_of_two():
    # pytest.raises 表示：这段代码应该抛出指定的异常。
    with pytest.raises(ValueError, match="at least 3"):
        JADE(sphere, [(-5, 5)], popsize=2)


def test_jade_can_run_with_population_of_three():
    optimizer = JADE(sphere, [(-5, 5)], popsize=3, max_nfev=9, seed=42)

    result = optimizer.run()

    assert result is not None
    # 初始化 3 次，再运行两代，每代 3 次；验证确实进入了变异和选择。
    assert result.nfev == 9
    assert result.generation == 2

def test_jade_prameters_scaled_to_bounds():
    optimizer = JADE(sphere, [(-5, 5)], popsize=50, seed=0)
    F, CR = optimizer.generate_params()
    assert np.all(F > 0) and np.all(F <= 1)
    assert np.all(CR >= 0) and np.all(CR <= 1)