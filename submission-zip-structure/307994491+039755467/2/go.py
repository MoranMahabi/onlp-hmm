import a2
from verification import bonus_driver_12
from verification.exercise_data_building.pos import build_POS_tagging_data


def go12():
    bonus_driver_12.model_driver_12(
        a2.Submission,
        build_POS_tagging_data(
            source_treebank_name="UD_English-EWT",
            git_hash="7be629932192bf1ceb35081fb29b8ecb0bd6d767"),
        passes=3)


go12()
