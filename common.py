"""Common functions shared by all scripts of this repository"""
from __future__ import annotations
import shutil
import zipfile
from typing import Optional, Any, Union, Callable, Iterable
import logging
import warnings
import os
from os.path import abspath, join, basename, isdir, isfile, dirname, splitext, getmtime
import stat
import sys
from datetime import datetime
from io import BytesIO
from dataclasses import dataclass
# third-party libs (require pip install):
import yaml
import h5py
import pandas as pd
import numpy as np
from numpy import ndarray
from tqdm import tqdm


# The program will stop if the successfully processed waveform ratio falls below this
# value that must be in [0, 1] (this makes spotting errors and checking log faster):
min_waveforms_ok_ratio = 1/100


# max discrepancy between PGA from catalog and computed PGA. A waveform is saved if:
# | PGA - PGA_computed | <= pga_retol * | PGA |
pga_retol = 1/4


@dataclass(frozen=True, slots=True)
class Waveform:
    """Simple class handling a Waveform (Time History single component)"""
    dt: float
    data: ndarray[float]


def main(py_script):  # noqa
    """main processing routine called from the command line"""

    # load functions
    accept_file: Callable = py_script.accept_file
    read_metadata: Callable = py_script.read_metadata
    find_sources: Callable = py_script.find_sources
    read_waveform: Callable = py_script.read_waveform
    post_process: Callable = py_script.post_process


    try:
        dataset_name, source_metadata_path, source_waveforms_path, dest_root_path = \
            read_script_args(sys.argv)
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"Source waveforms path: {source_waveforms_path}")
    print(f"Source metadata path:  {source_metadata_path}")

    dest_waveforms_path = join(dest_root_path, f'{dataset_name}.hdf')
    dest_aux_path =  join(dest_root_path, f'{dataset_name}')
    print(f"Dest. dataset path: {dest_waveforms_path}")
    print(f"Auxiliary data (e.g. logfile) will be put in: {dest_aux_path}")

    existing = isdir(dest_aux_path) or isfile(dest_waveforms_path)

    if existing:
        res = input(
            f'\nSome destination data already exists. Delete and re-create all? '
            f'(y: yes, Any key: quit)\n'
        )
        if res not in ('y', ):
            sys.exit(1)

    os.makedirs(dest_aux_path, exist_ok=True)

    dest_log_path = join(dest_aux_path, f"{dataset_name}.log")
    dest_metadata_path = join(dest_aux_path, f"{dataset_name}.meta.only.hdf")
    for fle in [dest_metadata_path, dest_waveforms_path, dest_log_path]:
        if isfile(fle):
            os.unlink(fle)  # not required for log (but it's easier to do it this way)
        assert not isfile(fle), f"Could not remove {fle}"
        fle_parent = dirname(fle)
        assert isdir(fle_parent), f"Parent dir does not exist: {fle_parent}"

    setup_logging(dest_log_path)
    logging.info(f'Working directory: {abspath(os.getcwd())}')
    logging.info(f'Run command      : {" ".join([sys.executable] + sys.argv)}')

    # Reading metadata fields dtypes and info:
    try:
        src_metadata_fields_path = join(dirname(__file__), 'metadata_fields.yml')
        metadata_fields = get_metadata_fields(src_metadata_fields_path)

        dest_metadata_fields_path = join(
            dest_root_path, basename(src_metadata_fields_path)
        )
        if not isfile(dest_metadata_fields_path) or (
            getmtime(dest_metadata_fields_path) < getmtime(src_metadata_fields_path)
        ):
            # cp and set modification time to NOW:
            shutil.copy(src_metadata_fields_path, dest_metadata_fields_path)

    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    mandatory_fields = [
        m for m in metadata_fields
        if '[mandatory]' in metadata_fields[m].get('help', '').lower()
    ]

    print(f'Scanning source waveforms directory...', end=" ", flush=True)
    files = {f for f in scan_dir(source_waveforms_path) if accept_file(f)}
    assert len(files), 'No files found'
    msg = f'{len(files):,} file(s) found'
    print(msg)
    logging.info(msg)

    print(f'Reading source metadata file...', end=" ", flush=True)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("once", pd.errors.DtypeWarning)
        metadata = read_metadata(source_metadata_path, files)
        if caught_warnings:
            for _w_ in caught_warnings:
                print(f'({_w_.message})', end=" ", flush=True)
    for _c in ['event_id', 'station_id']:
        assert isinstance(metadata[_c].dtype, pd.CategoricalDtype), \
            'event_id and station_id must be categorical'
        metadata_fields[_c]['dtype'] = metadata[_c].dtype

    msg = (
        f'{len(metadata):,} record(s), {len(metadata.columns):,} field(s) per record'
    )

    # csv_args: dict[str, Any] = dict(source_metadata_csv_args)
    # # csv_args.setdefault('chunksize', 10000)
    # csv_args.setdefault(
    #     'usecols', csv_args.get('usecols', {}) | source_metadata_fields.keys()
    # )
    # with warnings.catch_warnings(record=True) as _w_:
    #     warnings.simplefilter("always", pd.errors.DtypeWarning)
    #     metadata = pd.read_csv(source_metadata_path, **csv_args)
    #     if _w_:
    #         print(f'({_w_[0].message})', end=" ", flush=True)
    # metadata = metadata.rename(
    #     columns={k: v for k, v in source_metadata_fields.items() if v is not None}
    # )
    # old_len = len(metadata)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
    #     metadata = pre_process(metadata, source_metadata_path, files).copy()
    #
    # for col in ['event_id', 'station_id']:
    #     assert isinstance(metadata[col].dtype, pd.CategoricalDtype)
    #     metadata_fields[col]['dtype'] = metadata[col].dtype
    #
    # if len(metadata) < old_len:
    #     logging.warning(f'{old_len - len(metadata)} metadata row(s) '
    #                     f'removed in pre-processing stage')
    # msg = (f'{len(metadata):,} record(s), '
    #        f'{len(metadata.columns):,} field(s) per record, '
    #        f'{old_len - len(metadata)} row(s) removed in pre-process')
    print(msg)
    logging.info(msg)

    print(f'Creating harmonized dataset from source')
    pbar = tqdm(
        total=len(files),
        bar_format="{percentage:3.0f}%|{bar} {postfix} | ~{remaining}s remaining"
    )
    records = []
    item_num = 0
    errs = 0
    saved_waveforms = 0
    waveforms_hdf_file = h5py.File(dest_waveforms_path, "w")
    while len(files):
        num_files = 1
        file = files.pop()
        waveforms_root_group = waveforms_hdf_file.require_group("waveforms")

        try:
            h1_path, h2_path, v_path, record = find_sources(file, metadata)

            # checks:
            if sum(_ is not None for _ in (h1_path, h2_path, v_path)) == 0:
                continue
            if not isinstance(record, pd.Series):
                raise Exception('No metadata record found (no pd.Series)')
            record = record.copy()
            for _ in (h1_path, h2_path, v_path):
                if _ in files:
                    num_files += 1
                    files.remove(_)

            comps = {}
            for cmp_name, cmp_path in zip(('h1', 'h2', 'v'), (h1_path, h2_path, v_path)):
                comps[cmp_name] = None
                if cmp_path:
                    try:
                        with open_file(cmp_path) as file_p:
                            comps[cmp_name] = read_waveform(cmp_path, file_p, record)
                    except OSError:
                        pass
            if all(_ is None for _ in comps.values()):
                raise Exception('No waveform read')
            if len(set(_.dt for _ in comps.values() if _ is not None)) != 1:
                raise Exception('Waveform components have mismatching dt')

            # process waveforms
            h1, h2, v = comps.get('h1'), comps.get('h2'), comps.get('v')
            # old_record = dict(record)  # for testing purposes
            new_record, h1, h2, v = post_process(record, h1, h2, v)

            # Assign code generated fields:
            item_num += 1
            new_record['id'] = item_num
            # finalize clean_record:
            avail_comps, sampling_rate = extract_metadata_from_waveforms(h1, h2, v)
            new_record['available_components'] = avail_comps
            new_record['sampling_rate'] = sampling_rate if \
                pd.notna(sampling_rate) else \
                int(metadata_fields['sampling_rate']['default'])

            clean_record = {}
            for f in metadata_fields:
                default_val = metadata_fields[f].get('default')
                dtype = metadata_fields[f]['dtype']
                if default_val is None:
                    if dtype == 'datetime':
                        default_val = pd.NaT
                    elif dtype == 'float':
                        default_val = np.nan
                val = new_record.get(f)
                try:
                    clean_record[f] = cast_dtype(val, dtype, default_val)
                    if f in mandatory_fields and pd.isna(clean_record[f]):
                        val = 'N/A'
                        raise AssertionError()
                except AssertionError:
                    raise AssertionError(f'Invalid value for "{f}": {str(val)}')

            # final checks:
            check_final_metadata(clean_record, h1, h2, v)

            # save waveforms
            saved_waveforms += 1
            save_waveforms(
                waveforms_root_group,
                # str(clean_record['id']),
                f"{clean_record['event_id']}/{clean_record['station_id']}",
                h1, h2, v
            )

            # append metadata record (save later, see below):
            records.append(clean_record)

        except Exception as exc:
            fname, lineno = exc_func_and_lineno(exc, __file__)
            logging.error(f"{exc}. File: {file}. Function {fname}, line {lineno}")
            errs += 1
        finally:
            pbar.set_postfix({"saved waveforms": f"{saved_waveforms:,}"})
            pbar.update(num_files)

        # save metadata:
        if len(records) > 1000:
            save_metadata(
                dest_metadata_path,
                pd.DataFrame(records),
                metadata_fields
            )
            records = []

        if pbar.n / pbar.total > (1 - min_waveforms_ok_ratio):
            # we processed enough data (1 - waveforms_ok_ratio)
            ok_ratio = 1 - (errs / pbar.total)
            if ok_ratio < min_waveforms_ok_ratio:
                # the processed data error ratio is too high:
                msg = f'Too many errors ({errs} of {pbar.total} records)'
                print(msg, file=sys.stderr)
                logging.error(msg)
                sys.exit(1)

    if len(records):
        save_metadata(dest_metadata_path, pd.DataFrame(records), metadata_fields)

    # put metadata int
    doc_grp = waveforms_hdf_file.require_group("metadata_doc")
    for col, c_data in metadata_fields.items():
        doc_grp.attrs[col] = f"[{c_data['dtype']}] {c_data['help']}"

    # write metadata in the file (with pandas this time):
    metadata: pd.DataFrame = pd.read_hdf(dest_metadata_path)  # noqa
    meta_changed = False
    for col in metadata_fields:
        values = metadata[col]
        if values.dtype.__class__.__name__ != 'CategoricalDtype':
            tmp_values = values.astype('category')
            if tmp_values.memory_usage(deep=True, index=False) <= \
                    0.5 * values.memory_usage(deep=True, index=False):
                metadata[col] = tmp_values
                meta_changed = True

    # close file:
    waveforms_hdf_file.flush()
    waveforms_hdf_file.close()

    # save metadata in waveforms file:
    metadata.to_hdf(
        dest_waveforms_path,
        key="metadata",  # name of the table in the HDF5 file
        format="table",
        mode="a"  # append mode; keeps existing groups/datasets
    )

    if meta_changed:
        # save back metadata (we compressed some columns):
        save_metadata(dest_metadata_path, metadata, force_overwrite=True)

    if isfile(dest_waveforms_path):
        os.chmod(
            dest_waveforms_path,
            os.stat(dest_waveforms_path).st_mode | stat.S_IRGRP | stat.S_IROTH
        )

    pbar.close()
    msg = f'Dataset created: {saved_waveforms:,} waveform(s) ' \
          f'(either already or newly created, depending on settings)'
    print(msg)
    logging.info(msg)
    sys.exit(0)


def read_script_args(sys_argv):

    if len(sys_argv) != 2:
        raise ValueError(f'Error: invalid argument, provide a valid yaml file')

    yaml_path = sys_argv[1]
    if not isfile(yaml_path):
        raise ValueError(f'Error: the file {yaml_path} does not exist')

    try:
        with open(yaml_path) as _:
            data = yaml.safe_load(_)
        assert isfile(data['source_metadata']), \
            f"'source_metadata' is not a file: {data['source_metadata']}"
        assert isdir(data['source_data']), \
            f"'source_data' is not a directory: {data['source_data']}"
        return (
            splitext(basename(yaml_path))[0],  ## dataset name
            data['source_metadata'],
            data['source_data'],
            data['destination']
        )
    except Exception as exc:
        raise ValueError(f'Yaml error ({basename(yaml_path)}): {exc}')


def setup_logging(filename):
    logger = logging.getLogger()  # root logger
    # if not logger.handlers:
    handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def scan_dir(source_root_dir) -> Iterable[str]:
    """Yields files by scan the given directory, recursively.
    Zip files are opened and treated as directories
    Use open_file to open any returned file path
    """
    for entry in os.scandir(source_root_dir):
        file_path = abspath(entry.path)

        if entry.is_dir():
            yield from scan_dir(file_path)
        elif splitext(entry.name)[1].lower() == '.zip':
            try:
                with zipfile.ZipFile(file_path, 'r') as z:
                    for name in z.namelist():
                        file_path2 = join(file_path, name)
                        yield file_path2
            except zipfile.BadZipFile as exc:
                logging.info(f'Skipping bad zip file: {file_path}')
                pass
        else:
            yield file_path


def open_file(file_path) -> BytesIO:
    """
    Open a regular file or a file inside a .zip archive. file_path is any item
    returned by `scan_dir`
    """
    fp_lower = file_path.lower()
    try:
        if f".zip{os.sep}" in fp_lower:
            idx = fp_lower.index(".zip")
            zip_path, inner_path = file_path[:idx + 4], file_path[idx + 5:]
            with zipfile.ZipFile(zip_path, "r") as z:
                data = z.read(inner_path)
        elif fp_lower.endswith(".zip"):
            zip_path = file_path
            with zipfile.ZipFile(zip_path, "r") as z:
                namelist = z.namelist()
                if len(namelist) != 1:
                    raise OSError(
                        f"{file_path} contains {len(namelist)} files, expected one only")
                data = z.read(namelist[0])
        else:
            with open(file_path, "rb") as f:
                data = f.read()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, KeyError) as e:
        raise OSError(f"Failed to read {file_path}: {e}") from e

    return BytesIO(data)


def get_metadata_fields(src_path):
    """
    Get the YAML metadat fields and saves it
    into dest_root_path. Returns the dict of the parsed YAML
    """
    with open(src_path, 'rb') as _:
        metadata_fields_content = _.read()
        # Load YAML into Python dict
        metadata_fields = yaml.safe_load(metadata_fields_content.decode("utf-8"))
        # convert dtypes:
        for m in metadata_fields:
            field = metadata_fields[m]
            field_dtype = field['dtype']
            if isinstance(field_dtype, (list, tuple)):
                assert 'default' not in field or field['default'] in field_dtype
                field['dtype'] = pd.CategoricalDtype(field_dtype)
    return metadata_fields


def cast_dtype(val: Any, dtype: Union[str, pd.CategoricalDtype], default_val: Any):
    if pd.isna(val) or not str(val).strip():
        val = default_val
    if dtype == 'int':
        assert isinstance(val, int) or (isinstance(val, float) and int(val) == val)
        val = int(val)
    elif dtype == 'bool':
        if val in {0, 1}:
            val = bool(val)
        assert isinstance(val, bool)
    elif val is not None:
        if dtype == 'datetime':
            if hasattr(val, 'to_pydatetime'):  # for safety
                val = val.to_pydatetime()
            elif isinstance(val, str):
                val = datetime.fromisoformat(val)
            assert isinstance(val, datetime)
        elif dtype == 'str':
            assert isinstance(val, str)
        elif dtype == 'float':
            if not isinstance(val, float):
                val = float(val)
        elif isinstance(dtype, pd.CategoricalDtype):
            assert val in dtype.categories
    return val


def check_final_metadata(
    metadata: dict,
    h1: Optional[Waveform],
    h2: Optional[Waveform],
    v: Optional[Waveform]
):
    pga = np.abs(metadata['PGA'])
    if pga < 0:
        pga = metadata['PGA'] = abs(pga)
    pga_c = None
    if h1 is not None and h2 is not None:
        pga_c = np.sqrt(np.max(np.abs(h1.data)) * np.max(np.abs(h2.data)))
    elif h1 is not None and h2 is None:
        pga_c = np.max(np.abs(h1.data))
    elif h1 is None and h2 is not None:
        pga_c = np.max(np.abs(h2.data))
    else:
        pga_c = np.max(np.abs(v.data))
    if pga_c is not None:
        rtol = pga_retol
        assert np.isclose(pga_c, pga, rtol=rtol, atol=0), \
            f"|PGA - PGA_computed|={int(100 * (pga - pga_c) / pga)}%PGA"


def extract_metadata_from_waveforms(
    h1: Optional[Waveform],
    h2: Optional[Waveform],
    v: Optional[Waveform]
) -> tuple[Optional[str], Optional[int]]:
    dt = None
    avail_comps = ''
    for comp, avail_comp_str in zip((h1, h2, v), ('H', 'H', 'V')):
        if comp is None:
            continue
        if avail_comps == '':  # first non null component
            dt = comp.dt
        elif comp.dt != dt:
            dt = None
        avail_comps += avail_comp_str

    sampling_rate = int(1./dt) if dt is not None and int(1./dt) == 1./dt else None
    return avail_comps, sampling_rate


def save_metadata(
    dest_metadata_path: str,
    metadata: pd.DataFrame,
    metadata_fields: Optional[dict] = None,
    force_overwrite=False
):
    if metadata is None or metadata.empty:
        return

    if metadata_fields is not None:
        # check fields
        new_metadata_df = pd.DataFrame(metadata)
        for col in metadata_fields:
            dtype = metadata_fields[col]['dtype']
            if dtype == 'str':
                raise ValueError('Invalid `str` dtype')
            elif dtype == 'datetime':
                new_metadata_df[col] = pd.to_datetime(new_metadata_df[col])
            else:
                new_metadata_df[col] = new_metadata_df[col].astype(dtype)

            new_metadata_df = new_metadata_df[list(metadata_fields)]
    else:
        new_metadata_df = metadata

    hdf_kwargs = {
        'key': "metadata",  # table name
        'mode': "w" if force_overwrite else "a",
        # (1st time creates a new file because we deleted it, see above)
        'format': "table",  # required for appendable table
        'append': not force_overwrite,  # first batch still uses append=True
        'index': False,
        # 'min_itemsize': {
        #     'event_id': metadata["event_id"].str.len,
        #     'station_id': metadata["station_id"].str.len,
        # },  # required for strings (used only the 1st time to_hdf is called)  # noqa
        # 'data_columns': [],  # list of columns you need to query these later
    }
    new_metadata_df.to_hdf(dest_metadata_path, **hdf_kwargs)


def save_waveforms(
    h5_file: Union[h5py.Group, h5py.File],
    path: str,
    h1: Optional[Waveform],
    h2: Optional[Waveform],
    v: Optional[Waveform]
):

    dts = {x.dt for x in (h1, h2, v) if x is not None}
    assert len(dts), "No waveform to save"  # safety check
    data_has_samples = {len(x.data) for x in (h1, h2, v) if x is not None}
    assert all(data_has_samples), "Cannot save empty waveform(s)"

    assert len(dts) == 1, "Non-unique dt in waveforms"
    dt = dts.pop() if dts else None

    paths = path.split('/')
    h5_group = h5_file
    for _ in paths[:-1]:
        h5_group = h5_group.require_group(_)

    # Create variable-length array dataset
    empty = np.array([], dtype=float)
    dset = h5_group.create_dataset(paths[-1], (3,), dtype=h5py.vlen_dtype(np.float64))
    dset[0] = h1.data if h1 is not None else empty
    dset[1] = h2.data if h2 is not None else empty
    dset[2] = v.data if v is not None else empty
    # Store dt as attribute
    dset.attrs['dt'] = dt


def exc_func_and_lineno(exc, module_path: str = __file__) -> tuple[str, int]:
    """
    Return the innermost function name and line number within `__file__`
    that raised `exc`
    """
    tb = exc.__traceback__ if isfile(module_path) else None
    deepest = None

    while tb:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename

        if isfile(filename) and os.path.samefile(filename, module_path):
            deepest = frame

        tb = tb.tb_next

    # fallback to outermost frame if none found
    if deepest is None:
        deepest = exc.__traceback__.tb_frame

    return deepest.f_code.co_name, deepest.f_lineno


