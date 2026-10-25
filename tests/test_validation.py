import numpy as np
import pytest

from delab.de import DE


@pytest.mark.parametrize(
    "overrides, error, message",
    [
        ({"func": None}, TypeError, "func must be callable"),
        ({"bounds": []}, ValueError, "bounds"),
        ({"bounds": [-5, 5]}, ValueError, "bounds"),
        ({"bounds": [(-5, 5, 10)]}, ValueError, "bounds"),
        ({"bounds": [(-5, 5), (0,)]}, ValueError, "bounds"),
        ({"bounds": [("bad", 5)]}, ValueError, "bounds"),
        ({"bounds": [(5, -5)]}, ValueError, "lower.*upper"),
        ({"bounds": [(0, np.inf)]}, ValueError, "finite"),
        ({"bounds": [(np.nan, 5)]}, ValueError, "finite"),
        ({"popsize": True}, TypeError, "popsize.*integer"),
        ({"popsize": 4.5}, TypeError, "popsize.*integer"),
        ({"popsize": 0}, ValueError, "popsize.*positive"),
        ({"popsize": -1}, ValueError, "popsize.*positive"),
        ({"popsize": 3}, ValueError, "popsize.*at least 4"),
        ({"max_nfev": True}, TypeError, "max_nfev.*integer"),
        ({"max_nfev": 8.5}, TypeError, "max_nfev.*integer"),
        ({"max_nfev": 0}, ValueError, "max_nfev.*positive"),
        ({"max_nfev": 3}, ValueError, "max_nfev.*at least popsize"),
        ({"F": "0.5"}, TypeError, "F.*real number"),
        ({"F": True}, TypeError, "F.*real number"),
        ({"F": 0}, ValueError, "F.*positive"),
        ({"F": -0.5}, ValueError, "F.*positive"),
        ({"F": np.nan}, ValueError, "F.*finite"),
        ({"F": np.inf}, ValueError, "F.*finite"),
        ({"CR": [0.5]}, TypeError, "CR.*real number"),
        ({"CR": np.bool_(True)}, TypeError, "CR.*real number"),
        ({"CR": -0.1}, ValueError, "CR.*between 0 and 1"),
        ({"CR": 1.1}, ValueError, "CR.*between 0 and 1"),
        ({"CR": np.nan}, ValueError, "CR.*finite"),
        ({"CR": np.inf}, ValueError, "CR.*finite"),
    ],
)
def test_invalid_parameters_fail_at_construction(overrides, error, message):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return float(np.sum(x**2))

    kwargs = dict(func=objective, bounds=[(-5, 5)], popsize=4, max_nfev=8)
    kwargs.update(overrides)

    # 构造时就要报错，无需调用 run()，也不应花费目标函数评估次数。
    with pytest.raises(error, match=message):
        DE(**kwargs)
    assert calls == []


@pytest.mark.parametrize("CR", [0.0, 1.0])
def test_valid_edge_parameters_can_run(CR):
    def objective(x):
        assert x[0] == 2.0  # 相同上下界表示这个变量固定不变。
        return float(np.sum(x**2))

    optimizer = DE(
        objective,
        bounds=[(2, 2), (-5, 5)],
        popsize=np.int64(4),
        max_nfev=np.int64(11),
        F=np.float64(1.5),
        CR=CR,
        seed=0,
    )
    result = optimizer.run()
    assert result.nfev == 8  # 剩余 3 次不足以评估完整的 4 个个体。
    assert result.generation == 1


def test_exhausted_budget_is_checked_before_more_work(monkeypatch):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return float(np.sum(x**2))

    optimizer = DE(objective, [(-5, 5)], popsize=4, max_nfev=4, seed=0)
    optimizer.run()

    def unexpected_generate_params():
        pytest.fail("An exhausted step must stop before generating parameters.")

    monkeypatch.setattr(optimizer, "generate_params", unexpected_generate_params)
    with pytest.raises(ValueError, match="Maximum number"):
        optimizer.step()
    with pytest.raises(ValueError, match="Maximum number"):
        optimizer.evaluate(np.array([[0.0]]))
    assert optimizer.nfev == len(calls) == 4
