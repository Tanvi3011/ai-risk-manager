from src.config import (
    NUM_TRANSACTIONS,
    NUM_USERS,
    RANDOM_SEED
)


def test_dataset_configuration():

    assert NUM_TRANSACTIONS > 0
    assert NUM_USERS > 0
    assert RANDOM_SEED == 42