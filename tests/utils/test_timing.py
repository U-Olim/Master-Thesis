"""Tests for lightweight runtime profiling helpers."""

import math
import time

import numpy as np

from utils.timing import RUNTIME_COLUMNS, RuntimeProfile, empty_runtime_columns


def test_runtime_profile_records_positive_elapsed_time() -> None:
    profile = RuntimeProfile()

    with profile.time("stage"):
        time.sleep(0.001)

    assert profile.get("stage") > 0.0


def test_runtime_profile_repeated_name_accumulates() -> None:
    profile = RuntimeProfile()

    with profile.time("stage"):
        time.sleep(0.001)
    first = profile.get("stage")
    with profile.time("stage"):
        time.sleep(0.001)

    assert profile.get("stage") > first


def test_runtime_profile_missing_key_returns_nan() -> None:
    profile = RuntimeProfile()

    assert math.isnan(profile.get("missing"))


def test_empty_runtime_columns_returns_all_expected_columns() -> None:
    columns = empty_runtime_columns()

    assert set(columns) == set(RUNTIME_COLUMNS)
    assert all(np.isnan(value) for value in columns.values())
