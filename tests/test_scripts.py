import importlib
import importlib.util
import shutil
import os
from os.path import dirname, join, abspath, isdir, splitext, basename, isfile
from unittest.mock import patch
from urllib.request import urlopen

import yaml
import pytest
# if running in pycharm, I guess sys.path is inserted, so:
import sys
sys.path.append(dirname(dirname(__file__)))
import common

dest_data_dir = join(abspath(dirname(__file__)), 'tmp.datasets')


with urlopen('https://github.com/rizac/gmgt/gmgt.py') as _1:
    with open(join(dest_data_dir, 'gmgt.py'), 'w') as _2:
        _2.write(_1.read())

from gmgt import get_records  # noqa  (it will find it)

# import

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
        for h1, h2, v, dt, meta in get_records(dataset_path):
            # check we have all fields:
            assert set(meta._fields) - meta_fields == set()  # noqa
            # check waveforms have points in waveforms (if not empty):
            assert all(len(_) == 0 or len(_) > 1000 for _ in [h1, h2, v])
            # counter:
            zum += 1

        # records function check:
        filters_less_than = 0
        for k, v in [
            ('missing_', False),
            ('max_', 6),
            ('min_', 5.5),
            ('', 5.5),
            ('', [4.5, 5])
        ]:
            zum2 = 0
            for _ in get_records(dataset_path, **{k+'magnitude': v}):
                zum2 += 1
            assert zum2 <= zum
            if zum2 < zum:
                filters_less_than += 1
        assert filters_less_than >= 3

        with pytest.raises(Exception):
            for _ in get_records(dataset_path, min_station_id = 'a'):
                continue

        c = 0
        for _ in get_records(dataset_path, min_origin_time_resolution='D'):
            c += 1
        assert c > 0

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
