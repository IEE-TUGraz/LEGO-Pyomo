import markov
from LEGO.LEGOUtilities import MPSFileManager
from LEGO.helpers.CompareModels import ModelTypeForComparison, compareModels


def test_markovIllustrative(tmp_path):
    """
    Runs the Markov illustrative model and compares the generated MPS file to a reference MPS file.
    :return: None
    """
    markov.main("data/markov", no_sqlite=True, no_regret_plot=True, markov_light_only=True, save_mps=True)
    cases = ["Cyclic", "Markli", "NoEnf", "Truth "]

    output_mps = [f"{case}.mps" for case in cases]

    comparisonCommit = "f1f8f046447a0e8e6b4d8c01d62d1a9cf8bdbb5d"
    comparison_mps = [f"tests/data/mps-archive/markov-{case}-{comparisonCommit}.mps" for case in cases]

    with MPSFileManager(comparison_mps) as mps_files:
        for i, mps_file in enumerate(mps_files):
            assert compareModels(ModelTypeForComparison.MPS_FILE, output_mps[i], False,
                                 ModelTypeForComparison.MPS_FILE, mps_file, False,
                                 tmp_folder_path=tmp_path, print_additional_information=True)
