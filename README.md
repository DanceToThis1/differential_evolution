# de-lab

用于学习和实验的差分进化（Differential Evolution）项目，要求 Python 3.11 或更高版本。
目前实现了基础 DE 和个体参数自适应的 jDE：共享 rand/1 变异、二项交叉、
与边界取中点的越界修复和贪心选择。
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

## 变异策略

`src/delab/strategies.py` 保存独立的变异函数，目前提供 `rand_1(pop, F, rng)`。
`pop` 是形状为 `(n, dim)` 的 NumPy 数值数组，`F` 是长度为 `n` 的数组，
`rng` 是随机数生成器。调用者需要保证 `n >= 4`；通过 DE/jDE 调用时，构造函数会检查。

该函数为每个个体从其他个体中不放回抽取三个索引，计算
`pop[a] + F[i] * (pop[b] - pop[c])`，返回新数组，不修改种群和参数。
随机数生成器的状态会推进；函数只负责变异，越界修复和交叉仍由优化器完成。

`DE.mutate()` 把实例中的种群、参数和随机数生成器交给 `rand_1()`。
jDE 继承这个方法，因此也使用同一份策略实现。策略可以直接导入，无需构造优化器：

```python
from delab.strategies import rand_1
```

## jDE：每个个体保存自己的参数

```python
from delab import JDE

optimizer = JDE(sphere, bounds=[(-5, 5), (-5, 5)], popsize=30, max_nfev=5000, seed=42)
result = optimizer.run()
print(result.best_f)
print(optimizer.F_i, optimizer.CR_i)
```

这里的 `sphere` 沿用上面的目标函数。`JDE` 继承 `DE`，构造参数和验证规则相同，
其中 `F`、`CR` 指定每个个体的初始参数，默认分别为 0.5、0.9。
已保存的个体参数放在长度为 `popsize` 的 `F_i`、`CR_i` 数组里。

每代分别以 0.1 的概率为每个个体重新采样候选 `F`、`CR`，否则继承其已保存的值。
新的 `F` 均匀采样自 `[0.1, 1.0)`，新的 `CR` 均匀采样自 `[0, 1)`。
这些固定概率和采样范围采用 [Brest 等人的 jDE 论文](https://doi.org/10.1109/TEVC.2006.872133)
中的设置，区间写法遵循 NumPy 随机采样不含上端点的约定。

`generate_params()` 在副本上生成候选参数，试验个体通过选择后，
`adapt()` 才把对应候选写回。两种优化器都采用严格改进规则：
`trial_fitness < fitness` 才替换个体；相等时保留原个体，jDE 也保留其原参数。

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
评估点合法性、构造参数检查、评估预算、独立变异策略以及 jDE 的候选参数和成功保留规则。
只运行参数检查测试：

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

`src/delab/__init__.py` 显式定义包，并导出 `DE`、`JDE`、`Optimizer` 和 `GenerationInfo`。
