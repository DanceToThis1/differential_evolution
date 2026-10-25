# de-lab

用于学习和实验的差分进化（Differential Evolution）项目，要求 Python 3.11 或更高版本。
目前实现了基础 DE：rand/1 变异、二项交叉、与边界取中点的越界修复和贪心选择。
目标函数按最小化处理，每次接收一个一维 NumPy 数组，返回一个实数。

## 安装

在项目根目录运行（以下激活命令适用于 Linux/macOS）：

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e ".[dev]"
```

`-e` 表示可编辑安装，修改 `src/` 中的代码后无需重新安装。
`[dev]` 同时安装 pytest 和 Ruff，供测试与代码检查使用。
只使用优化器时可运行 `python -m pip install -e .`，运行依赖只有 NumPy。

## 使用

```python
import numpy as np

from delab import DE


def sphere(x):
    return float(np.sum(x**2))


optimizer = DE(
    func=sphere,
    bounds=[(-5, 5), (-5, 5)],
    popsize=30,
    max_nfev=5000,
    seed=42,
    F=0.5,
    CR=0.9,
)
result = optimizer.run()
print(result.best_x, result.best_f, result.nfev)
```

`run()` 返回最终的 `GenerationInfo`。若要观察每一代，可以改用
`for info in optimizer.iterate(): ...`；它也会产出初始化后的第 0 代。
每次重新运行实验应创建新的优化器实例，同一个实例只能启动一次运行。

## 参数约定

| 参数 | 要求 |
| --- | --- |
| `func` | 可调用对象，返回有限实数；返回值的有限性由调用者保证 |
| `bounds` | 非空的 `(dim, 2)` 数值数组或列表，边界有限且下界不大于上界；相等表示固定变量 |
| `popsize` | 整数，基础 DE/rand/1 至少为 4，默认 50 |
| `max_nfev` | 不小于 `popsize` 的正整数，默认 100000 |
| `F` | 有限正实数，默认 0.5 |
| `CR` | `[0, 1]` 内的有限实数，默认 0.9 |
| `seed` | 传给 `numpy.random.default_rng` 的随机种子，默认 `None` |

个数参数也接受 NumPy 整数，但不接受浮点数或布尔值；`F` 和 `CR` 也不接受布尔值。
构造参数不合法时立即报错：类型不匹配抛出 `TypeError`，取值不合法抛出 `ValueError`。
无法转换为数值边界数组的 `bounds` 抛出 `ValueError`。

初始化消耗 `popsize` 次评估，每代再消耗 `popsize` 次。
剩余预算不足一整代时停止，因此 `nfev` 可能小于 `max_nfev`。
直接调用 `step()` 或 `evaluate()` 仍会检查剩余预算，超出时抛出 `ValueError`。
`step()` 需要先通过 `iterate()` 初始化种群。

## 测试与代码检查

```bash
python -m pytest -q
python -m pytest --collect-only -q
python -m ruff check .
python -m ruff format --check .
```

pytest 从 `tests/` 收集测试。现有测试覆盖 Sphere 收敛、seed 可复现、越界修复、
评估点合法性、构造参数检查和评估预算。只运行参数检查测试：

```bash
python -m pytest tests/test_validation.py -v
```

`pytest.mark.parametrize` 用多组输入重复执行同一测试；
`pytest.raises` 验证代码是否抛出预期异常。

Ruff 检查常见代码错误和导入顺序；`python -m ruff format .` 自动统一格式。

## 项目配置

`pyproject.toml` 中：

- `[build-system]` 指定 setuptools 构建后端及其构建依赖。
- `[project]` 声明项目元数据和运行依赖。
- `[project.optional-dependencies]` 声明可选的 `dev` 开发依赖。
- `[tool.setuptools.packages.find]` 指定从 `src/` 发现 Python 包。
- `[tool.pytest.ini_options]` 指定测试目录。
- `[tool.ruff]` 和 `[tool.ruff.lint]` 配置格式和检查规则。

`src/delab/__init__.py` 显式定义包，并导出 `DE`、`Optimizer` 和 `GenerationInfo`。
