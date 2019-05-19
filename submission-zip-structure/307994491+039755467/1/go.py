import a1

from verification.exercise_data_building.pos import build_POS_tagging_data
from verification.model_driver_12 import model_driver_12


def go12():
    model_driver_12(
        a1.Submission,
        build_POS_tagging_data(
            source_treebank_name="UD_English-EWT",
            git_hash="7be629932192bf1ceb35081fb29b8ecb0bd6d767"),
        passes=3)


go12()
