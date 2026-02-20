import importlib
import importlib.util
import shutil
import os
from os.path import dirname, join, abspath, isdir, splitext, basename, isfile
from unittest.mock import patch
import yaml
import pytest
# if running in pycharm, I guess sys.path is inserted, so:
import sys
sys.path.append(dirname(dirname(__file__)))
import common

dest_data_dir = join(abspath(dirname(__file__)), 'tmp.datasets')


def tearDown():
    shutil.rmtree(dest_data_dir)
    # sys.path.pop(0)


def setUp():
    if not isdir(dest_data_dir):
        os.makedirs(dest_data_dir)
    # src = dirname(dirname(__file__))
    # shutil.copy(join(src, 'create_ngawest_dataset.py'),
    #             join(root, 'create_ngawest_dataset.py'))
    # sys.path.insert(0, root)


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    setUp()
    yield
    tearDown()


def run_(module_name:str, dataset: str, src_metadata_path: str, src_data_dir: str):
    # dataset is ngawest, esm, kik knet
    src_data_dir = os.path.expanduser(src_data_dir)
    assert isdir(src_data_dir)
    src_metadata_path = os.path.expanduser(src_metadata_path)
    assert isfile(src_metadata_path)

    config_path = join(dest_data_dir, f"{dataset}.yml")

    with open(config_path, 'w') as _:
        _.write(f"""
source_metadata: "{src_metadata_path}"
source_data: "{src_data_dir}"
destination: "{dest_data_dir}"
""")
    try:
        # project_root = abspath(dirname(dirname(__file__)))
        # os.chdir(project_root)
        # result = subprocess.run(
        #     [sys.executable, f'create_{dataset}_dataset.py', config_path],
        #     capture_output=True,
        #     text=True,
        #     check=True
        # )
        # stdout_text = result.stdout
        # stderr_text = result.stderr
        # asd = 9

        module_path = \
            abspath(join(dirname(dirname(__file__)), f'{module_name}.py'))
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        sys.argv = [f'{module_name}.py', config_path]
        # with patch(f"{module_name}.min_waveforms_ok_ratio", 0):
        with patch(f"common.min_waveforms_ok_ratio", 0):
            common.main(sys.modules[module_name])
    except SystemExit as err:
        assert err.args[0] == 0  # ok exit

        dataset_path = join(dest_data_dir, dataset + '.hdf')
        data = {
            'metadata': 0,
            'waveforms': 0,
            'metadata_doc': 0
        }
        with h5py.File(dataset_path, 'r') as f:
            assert sorted(f.keys()) == ['metadata', 'metadata_doc', 'waveforms']
            # keys = f['waveforms'].keys()
            # asd = 9
        sizes = hdf_dataset_sizes(dataset_path)
        assert sizes['metadata'] > 150000
        assert sizes['waveforms'] > 10

        assert isfile(join(dest_data_dir, dataset, f'{dataset}.meta.only.hdf'))
        assert isfile(join(dest_data_dir, dataset, dataset + '.log'))

        with open(join(dirname(dirname(__file__)), 'metadata_fields.yml'), 'rb') as f:
            meta_fields = set(yaml.safe_load(f).keys())

        # data check:
        zum = 0
        for h1, h2, v, dt, meta in records(dataset_path):
            # check we have all fields:
            assert set(meta._fields) - meta_fields == set()
            # check waveforms have points in waveforms (if not empty):
            assert all(len(_) == 0 or len(_) > 1000 for _ in [h1, h2, v])
            # counter:
            zum += 1

        # records function check:
        filters_less_than = 0
        for k, v in [
            ('missing_', False), ('max_', 6), ('min_', 5.5), ('', 5.5), ('', [4.5, 5])
        ]:
            zum2 = 0
            for _ in records(dataset_path, **{k+'magnitude': v}):
                zum2 += 1
            assert zum2 <= zum
            if zum2 < zum:
                filters_less_than += 1
        assert filters_less_than >= 3

    except Exception as e:
        # Raise a new exception with the subprocess traceback
        raise
    finally:
        del sys.modules[module_name]


def hdf_dataset_sizes(file_path):
    sizes = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            key = name.split("/")[0]
            if key != 'metadata':
                asd = 9
            sizes.setdefault(key, 0)
            sizes[key] += obj.size * obj.dtype.itemsize  # number of elements * bytes per element

    with h5py.File(file_path, 'r') as f:
        f.visititems(visit)

    return sizes


def test_ngawest2():
    run_(
        'create_ngawest2_dataset',
        'ngawest2',
        '~/Nextcloud/gmgt/source_data/ngawest2/Updated_NGA_West2_Flatfile_RotD50_d005_public_version.csv',
        '~/Nextcloud/gmgt/gmgt-collect-test-data/ngawest2'
    )

def test_kik():
    run_(
        'create_kik_knet_dataset',
        'kik',
        '~/Nextcloud/gmgt/source_data/kik/2025-001_Loviknes-et-al_1997_2024_kik_Oct24META.csv',
        "~/Nextcloud/gmgt/gmgt-collect-test-data/kik"
    )

def test_knet():
    run_(
        'create_kik_knet_dataset',
        'knet',
        '~/Nextcloud/gmgt/source_data/knet/2025-001_Loviknes-et-al_1996_2024_knet_Oct24META.csv',
        "~/Nextcloud/gmgt/gmgt-collect-test-data/knet"
    )

def test_esm():
    run_(
        'create_esm_dataset',
        'esm',
        '~/Nextcloud/gmgt/source_data/esm/ESM_flatfile_SA.csv',
        '~/Nextcloud/gmgt/gmgt-collect-test-data/esm'
    )


def tst_source_metadata_stats():
    import pandas as pd
    root = join(dirname(__file__), 'source_data')
    for file, sep in [
        ('esm/ESM_flatfile_SA.csv', ";"),
        ('ngawest2/ngawest2_metadata.csv', ','),
        ('kiknet_knet/kiknet_knet_metadata.csv', ",")
    ]:
        df = pd.read_csv(join(root, file), sep=sep)
        ratios = []
        for col in df.columns:
            notna = (~df[col].isin({-999, -999999, '-999', '-999999'})) & df[col].notna()
            ratios.append((col, int(100 * notna.sum() / len(df))))
        # Sort descending by ratio
        ratios_sorted = sorted(ratios, key=lambda x: x[1], reverse=True)

        print("\n")
        print(basename(file))
        for k in ratios_sorted:
            if k[1] > 95:
                print(k)
        print("\n\n")


import numpy as np
import pandas as pd
import h5py
from typing import Iterable, Optional


def records(
    hdf_path, **filters
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple]]:
    """
    Yield: (h1, h2, v, dt, metadata) for each matching record. The first three elements
    denote the record data (time history) and are numpy arrays of acceleration in m/s**2
    (first two horizontal and vertical component, respectively): empty arrays mean the
    component is not available. `dt` is the float denoting the data sampling interval
    (in s), and metadata is quite self-explanatory (you can access all metadata as
    normal attributes, e.g. `metadata.magnitude`)

    :param hdf_path: dataset path (HDF format) path with waveforms and metadata
    :param filters: a keyword argument whose parameters are any metadata fields,
        optionally prefixed with 'min_', 'max_' and 'missing_' mapped to a matching
        values in order to filter specific metadata row and yield only the corresponding
        data. Values cannot be None / nan, NaT: to get those values, use the 'missing_'
        prefix:, e.g. type `missing_magnitude: False` to yield only records where the
        magnitude is provided (not N/A). Values can also be list/tuples, in this case
        records whose fields are equal to any value in the list/tuple will be yielded

        Examples:
            for h1, h2, v, dt, m in records(path, min_magnitude=6):
            for h1, h2, v, dt, m in records(path, max_magnitude=6)
            for h1, h2, v, dt, m in records(path, magnitude=6)
            for h1, h2, v, dt, m in records(path, magnitude=[4, 5, 6])
    """
    chunk_size = 100000  # chunk for pandas read_hdf

    with pd.HDFStore(hdf_path, "r") as pd_f, h5py.File(hdf_path, "r") as h5_f:
        h5_root_group = h5_f["waveforms"]
        for chunk in pd_f.select("metadata", chunksize=chunk_size):  # noqa
            mask = pd.Series(True, index=chunk.index)
            for expr, value in filters.items():
                try:
                    if expr.startswith('min_'):
                        col = expr[4:]
                        # categorical column, need to work on categories:
                        if isinstance(chunk[col].dtype, pd.CategoricalDtype):
                            categs = chunk[col].cat.categories  # pandas Index
                            col_mask = chunk[col].isin(categs[categs >= value])
                        else:
                            col_mask = chunk[col] >= value
                    elif expr.startswith('max_'):
                        col = expr[4:]
                        if isinstance(chunk[col].dtype, pd.CategoricalDtype):
                            categs = chunk[col].cat.categories  # pandas Index
                            col_mask = chunk[col].isin(categs[categs <= value])
                        else:
                            col_mask = chunk[col] <= value
                    elif expr.startswith('missing_'):
                        col = expr[8:]
                        if value is True:
                            col_mask = chunk[col].isna()
                        elif value is False:
                            col_mask = chunk[col].notna()
                        else:
                            raise ValueError(f'True/False expected, found {value}')
                    else:
                        col = expr
                        if isinstance(value, (tuple, list, set)):
                            col_mask = chunk[col].isin(value)
                        else:
                            col_mask = chunk[col] == value
                    mask &= col_mask

                except (TypeError, ValueError, KeyError, AssertionError) as exc:
                    raise ValueError(f'Error in "{expr}": {exc}')

            for row in chunk[mask].itertuples(name='metadata', index=False):
                waveform = h5_root_group[row.event_id][row.station_id]
                yield waveform[0], waveform[1], waveform[2], waveform.attrs['dt'], row


def iter_records_v0(hdf_path, query_string=None, chunk_size=10000) -> \
        Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple]]:
    """
    THIS CODE IS A REMINDER OF HOW CODE COULD READ BACK THE DATA.

    Efficiently query metadata ON DISK and yield:
    (h1, h2, v, metadata_row) for each matching record.

    metadata_row is a Pandas Series.
    """
    with pd.HDFStore(hdf_path, "r") as pd_f, h5py.File(hdf_path, "r") as h5_f:
        h5_root_group = h5_f["waveforms"]
        for chunk in pd_f.select("metadata", where=query_string, chunksize=chunk_size):  # noqa
            for row in chunk.itertuples(index=False):
                waveform = h5_root_group[row.event_id][row.station_id]
                yield waveform[0], waveform[1], waveform[2], waveform.attrs['dt'], row
