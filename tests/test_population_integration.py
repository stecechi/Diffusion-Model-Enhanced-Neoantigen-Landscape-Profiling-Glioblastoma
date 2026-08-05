import numpy as np
from diffneoscore.population import Population, PopulationCoverage, panel_coverage


def test_population_equations_and_greedy_panel() -> None:
    populations = [Population("p1", "Asia", {"A*02:01": 0.2, "A*24:02": 0.3}), Population("p2", "Africa", {"A*02:01": 0.1, "B*53:01": 0.2})]
    calculator = PopulationCoverage(populations)
    first = calculator.target_coverage("first", ["A*02:01"])
    assert np.allclose(first.per_population, [0.36, 0.19])
    panel = calculator.greedy_panel({"first": ["A*02:01"], "second": ["A*24:02", "B*53:01"]}, 2)
    assert len(panel.targets) == 2
    assert panel.global_coverage > first.global_coverage
    matrix = np.stack((first.per_population, calculator.target_coverage("second", ["A*24:02", "B*53:01"]).per_population))
    assert np.allclose(panel_coverage(matrix), panel.per_population)
