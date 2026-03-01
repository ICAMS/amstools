import os
import pandas as pd

from amstools.utils import collect_raw_data

test_dirname = os.path.dirname(__file__)


def test_collect_raw_data_aims():
    fname = "collect_aims.pckl.gzip"
    if os.path.isfile(fname):
        os.remove(fname)
    assert not os.path.isfile(fname)
    collect_raw_data(fname, os.path.join(test_dirname, "aims"))
    assert os.path.isfile(fname)
    df = pd.read_pickle(fname, compression="gzip")
    print(df)
    assert len(df) == 3

    os.remove(fname)


def test_collect_raw_data_vasp():
    fname = "collect_vasp.pckl.gzip"
    if os.path.isfile(fname):
        os.remove(fname)
    assert not os.path.isfile(fname)

    collect_raw_data(fname, os.path.join(test_dirname, "restore_vasp_json"))
    assert os.path.isfile(fname)
    df = pd.read_pickle(fname, compression="gzip")
    print(df)
    assert len(df) == 1
    os.remove(fname)
