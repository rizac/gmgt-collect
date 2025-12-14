import importlib
import importlib.util
import shutil
import os
import sys
import h5py
from os.path import dirname, join, abspath, isdir, splitext, basename, isfile
from io import StringIO, BytesIO
import subprocess
from unittest.mock import patch

# import yaml
# import create_ngawest_dataset, create_kiknet_knet_dataset

import pytest

dest_data_dir = join(abspath(dirname(__file__)), 'tmp.datasets')
source_data_dir = join(abspath(dirname(__file__)), 'source_data')


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


def run_(dataset: str, metadata_file_name: str):
    # dataset is ngawest, esm, kinet_knet
    src_data_dir = join(source_data_dir, dataset)
    assert isdir(src_data_dir)
    src_metadata_path = join(src_data_dir, metadata_file_name)
    assert isfile(src_metadata_path)

    config_path = join(dest_data_dir, f"{dataset}.config.yml")

    with open(config_path, 'w') as _:
        _.write(f"""
source_metadata: "{src_metadata_path}"
source_data: "{src_data_dir}"
destination: "{dest_data_dir}"
""")
    module_name = f"create_{dataset}_dataset"
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
        with patch(f"{module_name}.min_waveforms_ok_ratio", 0):
            mod.main()
    except SystemExit as err:
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
        assert sizes['metadata'] > 250000
        assert sizes['waveforms'] > 250000

        assert err.args[0] == 0
        assert isfile(join(dest_data_dir, 'meta-only', dataset + '.hdf'))
        assert isfile(join(dest_data_dir, 'logs', dataset + '.log'))

        # assert isfile(w_file)
        # w_dir = join(dest_data_dir, 'waveforms')
        # assert isdir(w_dir)
        # some_file = False
        # for _, _, files in os.walk(w_dir):
        #     if files and any(splitext(_)[1] == '.h5' for _ in files):  # If `files` list is not empty
        #         some_file = True
        #         break  # Directory contains at least one file
        # assert some_file
        # # Get printed output
    except Exception as e:
        # Raise a new exception with the subprocess traceback
        raise
    finally:
        del sys.modules[module_name]
    #     # Restore originals
    #     sys.argv = original_argv
    #     sys.stdout = original_stdout


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


def test_nga_west2():
    run_('ngawest2', 'Updated_NGA_West2_Flatfile_RotD50_d005_public_version.csv')


def test_knet():
    run_('kiknet_knet', "2025-001_Loviknes-et-al_1996_2024_knet_Oct24META.csv")


def test_esm():
    run_('esm', 'ESM_flatfile_SA.csv')


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
