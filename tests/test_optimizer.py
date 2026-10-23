import numpy as np
from delab.de import DE

def sphere(x):
    return sum(value ** 2 for value in x)

def test_de_converges_on_sphere():
    optimizer = DE(func=sphere, bounds=[(-5, 5), (-5, 5)], popsize=30, max_nfev=5000, seed=42)
    result = optimizer.run()
    assert 0 <= sphere(result.best_x) < 1e-6
    np.testing.assert_allclose(result.best_f, sphere(result.best_x), rtol=0, atol=1e-12)
    assert optimizer.nfev <= 5000

def test_de_reproduces_results_with_same_seed():
    results = []
    for _ in range(2):
        optimizer = DE(func=sphere, bounds=[(-5, 5), (-5, 5)], popsize=30, max_nfev=5000, seed=42)
        results.append(optimizer.run())
    first, second = results
    np.testing.assert_array_equal(first.best_x, second.best_x)
    assert first.best_f == second.best_f
    assert first.nfev == second.nfev
    assert first.generation == second.generation
