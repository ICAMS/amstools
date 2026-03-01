import os.path

import numpy as np
from ase.atoms import Atoms
from ase.build import bulk

from amstools.utils import get_nearest_neighbor_distance, SQLiteStateDict


def test_get_nearest_neighbor_distance_periodic():
    bulk_atoms = bulk("Al", "fcc")
    nn_dist = get_nearest_neighbor_distance(bulk_atoms)
    print("nn_dist = ", nn_dist)
    assert nn_dist == 2.8637824638055176


def test_get_nearest_neighbor_distance_non_periodic():
    at = Atoms("Al2", positions=[[0, 0, 0], [0, 0, 1]], pbc=False)
    nn_dist = get_nearest_neighbor_distance(at)
    print("nn_dist = ", nn_dist)
    assert nn_dist == 1.0


def test_get_nearest_neighbor_distance_non_periodic_single_atom():
    at = Atoms("Al", positions=[[0, 0, 0]], pbc=False)
    nn_dist = get_nearest_neighbor_distance(at)
    print("nn_dist = ", nn_dist)
    assert nn_dist == np.inf


def test_SQLiteStateDict_read_write():
    test_db = "test_state_dict.db"
    try:
        if os.path.isfile(test_db):
            os.remove(test_db)
        sqlite = SQLiteStateDict(test_db)
        test_data = ["test_name", {"data_int": 1, "data_str": "2"}]
        sqlite.save_row(*test_data)
        res = sqlite["test_name"]
        print(res)
        assert res == test_data[1]
    finally:
        if os.path.isfile(test_db):
            os.remove(test_db)


def test_SQLiteStateDict_setitem():
    test_db = "test_state_dict.db"
    try:
        if os.path.isfile(test_db):
            os.remove(test_db)
        sqlite = SQLiteStateDict(test_db)
        test_data = ["test_name", {"data_int": 1, "data_str": "2"}]
        sqlite[test_data[0]] = test_data[1]
        res = sqlite["test_name"]
        print(res)
        assert res == test_data[1]
    finally:
        if os.path.isfile(test_db):
            os.remove(test_db)


def test_SQLiteStateDict_read_none():
    test_db = "test_state_dict.db"
    try:
        if os.path.isfile(test_db):
            os.remove(test_db)
        sqlite = SQLiteStateDict(test_db)
        res = sqlite["test_none_existing"]
        print(res)
        assert res == {}
    finally:
        if os.path.isfile(test_db):
            os.remove(test_db)


def test_SQLiteStateDict_contains():
    test_db = "test_state_dict.db"
    try:
        if os.path.isfile(test_db):
            os.remove(test_db)
        sqlite = SQLiteStateDict(test_db)
        test_data = ["test_name", {"data_int": 1, "data_str": "2"}]
        sqlite.save_row(*test_data)

        assert "test_name" in sqlite
        assert "test_name_non_exists" not in sqlite
    finally:
        if os.path.isfile(test_db):
            os.remove(test_db)


def test_SQLiteStateDict_len():
    test_db = "test_state_dict.db"
    try:
        if os.path.isfile(test_db):
            os.remove(test_db)
        sqlite = SQLiteStateDict(test_db)
        assert len(sqlite) == 0
        sqlite.save_row("test_name", {"data_int": 1, "data_str": "2"})
        assert len(sqlite) == 1
        sqlite.save_row("test_name2", {"data_int": 3, "data_str": "4"})
        assert len(sqlite) == 2
    finally:
        if os.path.isfile(test_db):
            os.remove(test_db)


def test_SQLiteStateDict_analyze_stats():

    try:
        test_db = "test_state_dict.db"
        if os.path.isfile(test_db):
            os.remove(test_db)
        sqlite = SQLiteStateDict(test_db)

        sqlite.save_row("1", {"status": "submitted", "data_str": "2"})
        sqlite.save_row("2", {"status": "finished", "data_str": "2"})
        sqlite.save_row("3", {"status": "error", "data_str": "2"})
        sqlite.save_row("4", {"no_status": "error", "data_str": "2"})
        sqlite.save_row("-1", {"status": "finished", "data_str": "2"})
        ht_names = set(["1", "2", "3", "4"])
        state_counter = sqlite.analyze_stats(ht_names)
        print(state_counter)
        assert sorted(state_counter) == sorted(
            {"submitted": 1, "finished": 1, "error": 1}
        )
    finally:
        if os.path.isfile(test_db):
            os.remove(test_db)


def test_SQLiteStateDict_has_not_finished_or_error_states():
    try:
        test_db = "test_state_dict.db"
        if os.path.isfile(test_db):
            os.remove(test_db)

        sqlite = SQLiteStateDict(test_db)

        sqlite.save_row("1", {"status": "submitted", "data_str": "2"})
        sqlite.save_row("2", {"status": "finished", "data_str": "2"})
        sqlite.save_row("3", {"status": "error", "data_str": "2"})
        sqlite.save_row("4", {"status": "error", "data_str": "2"})
        sqlite.save_row("5", {"status": "running", "data_str": "2"})
        ht_names = set(["1", "2", "3", "4"])
        has_not_finished_or_error_states = sqlite.has_not_finished_or_error_states(
            ht_names
        )

        assert has_not_finished_or_error_states
    finally:
        if os.path.isfile(test_db):
            os.remove(test_db)
