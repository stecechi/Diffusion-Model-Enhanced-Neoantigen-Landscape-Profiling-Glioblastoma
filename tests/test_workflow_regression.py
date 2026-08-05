from diffneoscore.population import Population, PopulationCoverage
from diffneoscore.workflow import AlleleScore, NeoantigenWorkflow, PeptideCandidate, peptide_windows


def test_peptide_generation_and_filtering_regression() -> None:
    wildtype = "MKTAYIAKQRQISFVKSHFSRQDILD"
    mutant = wildtype[:12] + "A" + wildtype[13:]
    windows = peptide_windows(wildtype, mutant, 12)
    assert {len(item[0]) for item in windows} == {8, 9, 10, 11}
    population = Population("reference", "Europe", {"A*02:01": 0.25})
    workflow = NeoantigenWorkflow(PopulationCoverage([population]))
    candidate = PeptideCandidate("v1", "TP53", windows[0][0], windows[0][1], windows[0][2], 2.0, 0.1, 0.9)
    scores = [AlleleScore("A*02:01", 100.0, 800.0, 0.8)]
    ranking = workflow.rank({candidate: scores})
    assert len(ranking) == 1
    assert ranking[0].global_coverage == 0.4375
    assert workflow.treatment_resistant(ranking[0])
