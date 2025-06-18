from CompareModels import compareModels, ModelTypeForComparison
from InOutModule.printer import Printer

printer = Printer.getInstance()

def test_deterministicVsExtensiveWithNoScenarios(tmp_path):
    mps_equal = compareModels(ModelTypeForComparison.DETERMINISTIC, "../data/example", True,
                            ModelTypeForComparison.EXTENSIVE_FORM, "../data/example", True)

    assert mps_equal

def test_simpleVsExtensiveWithTwoEqualScenarios(tmp_path):
    # TODO: Build a two-scenario case-study from example data
    printer.warning("Test not implemented yet!")

    # result = compareModels(ModelTypeForComparison.DETERMINISTIC, "../data/example",
    #              ModelTypeForComparison.EXTENSIVE_FORM, tmp_path)

    assert True


def test_stochasticSolutionWithTwoDifferentScenarios():
    printer.warning("Test not implemented yet!")
    assert True