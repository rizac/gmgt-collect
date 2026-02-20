"""
Python module to be executed as script to generate a new harmonized dataset from
heterogeneous sources. For a new dataset, copy rename this or any look-alike modules in
this directory and modify the editable part (see below). More details in README.md
"""
from __future__ import annotations

from typing import Optional
from os.path import join, basename, dirname, splitext
from datetime import datetime
from io import BytesIO
# third-party libs (require pip install):
import pandas as pd
import numpy as np
# scripts common module:
from common import Waveform


def accept_file(file_path) -> bool:
    """Tell whether the given source file can be accepted as waveform file

    :param file_path: the scanned file absolute path (it can also be a file within a zip
        file, in that case the parent directory name is the zip file name)
    """
    return splitext(file_path)[1] in {
        '.UD1', '.NS1', '.EW1', '.UD2', '.NS2', '.EW2', '.UD', '.NS', '.EW'
    }  # with *1 => borehole


# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {
    'skiprows': 3,
    # 'header': None  # for CSVs with no header
    # 'dtype': {}  # NOT RECOMMENDED, see `metadata_fields.yml` instead
    # 'usecols': []  # NOT RECOMMENDED, see `source_metadata_fields` below instead
}

# Mapping from source metadata columns to their new names. Map to None to skip renaming
# and just load the column data
source_metadata_fields = {
    'EQ_Code': 'event_id',
    "StationCode": 'station_id',

    'Origin_Meta': 'origin_time',
    'new_record_start_UTC': 'start_time',
    # 'RecordTime': None,
    'tP_JMA': 'p_wave_arrival_time',
    'tS_JMA': 's_wave_arrival_time',
    'PGA_EW': None,
    'PGA_NS': None,
    'PGA_rotd50': 'PGA',
    # 'azimuth': metadata.get(54),
    'Repi': 'epicentral_distance',
    'Rhypo': 'hypocentral_distance',
    'RJB_0': 'joyner_boore_distance',
    'RJB_1': 'joyner_boore_distance2',
    'Rrup_0': 'rupture_distance',
    'Rrup_1': 'rupture_distance2',
    # 'fault_normal_distance': None,
    "evLat._Meta": 'event_latitude',
    "evLong._Meta": 'event_longitude',
    "Depth. (km)_Meta": 'event_depth',
    "Mag._Meta": 'magnitude',
    "JMA_Magtype": 'magnitude_type',
    # 'depth_to_top_of_fault_rupture': None
    # 'fault_rupture_width': None,
    "fnet_Strike_0": 'strike',
    "fnet_Dip_0": 'dip',
    "fnet_Rake_0": 'rake',
    "fnet_Strike_1": 'strike2',
    "fnet_Dip_1": 'dip2',
    "fnet_Rake_1": 'rake2',
    "Focal_mechanism_BA": 'fault_type',
    # "vs30": "vs30",
    # "vs30measured": "vs30measured",
    'StationLat.': "station_latitude",
    'StationLong.': "station_longitude",
    'StationHeight(m)': "station_height",
    # "z1": "z1",
    # "z2pt5": "z2pt5",

    "fc0": "lower_cutoff_frequency_h1",  # FIXME CHECK THIS  hp_h1
    # "fc0": "lower_cutoff_frequency_h2",
    "fc1": "upper_cutoff_frequency_h1",
    # "fc1": "upper_cutoff_frequency_h2",
    # "fc0": "lowest_usable_frequency_h1",
    # "fc1": "lowest_usable_frequency_h2",  # if not sure, leave None
}


def pre_process(
    metadata: pd.DataFrame, metadata_path: str, files: set[str]
) -> pd.DataFrame:
    """Pre-process the metadata Dataframe. This is usually the place where the given
    dataframe is setup in order to easily find records from file names, or optimize
    some column data (e.g. convert strings to categorical).

    :param metadata: the metadata DataFrame. The DataFrame columns come from the global
        `source_metadata_fields` dict, using each value if not None, otherwise its key.
    :param metadata_path: the file path of the metadata DataFrame
    :param files: a set of file paths as returned from `scan_dir`

    :return: a pandas DataFrame optionally modified from `metadata`
    """
    metadata = metadata.dropna(subset=['event_id', 'station_id'])
    metadata['event_id'] = metadata['event_id'].astype(str).astype('category')
    # add categorical according to stations ".1" or ".2" (borehole for kik):
    metadata['station_id'] = metadata['station_id'].astype(str)
    sta_categs = []
    for s in pd.unique(metadata['station_id']):
        sta_categs.extend((s, s+ '.1', s+ '.2'))
    metadata['station_id'] = metadata['station_id'].astype(
        pd.api.types.CategoricalDtype(categories=sta_categs)
    )

    # add station metadata from separate CSV:
    sta_df = pd.read_csv(join(dirname(metadata_path), "Site Database of K-NET and "
                                                      "KiK-net Strong-Motion "
                                                      "Stations_v1.0.0_20201201_vs30."
                                                      "csv"), skiprows=2)
    # remove 1st row
    sta_df = sta_df.iloc[1:, :]
    # we assume that first vs30 is measured, second is inferred (looking at the csv):
    sta_df['vs30measured'] = sta_df['Vs30 measured'].astype(bool)
    sta_df['ZP1.0'] = sta_df['ZP1.0'].astype(float)
    sta_df['ZP2.5'] = sta_df['ZP2.5'].astype(float)
    assert sta_df['vs30measured'].any() and (~sta_df['vs30measured']).any()
    assert sta_df.columns[1] == 'VS30' and sta_df.columns[2] == 'VS30  '
    sta_df['vs30'] = sta_df['VS30']
    sta_df.loc[~sta_df['vs30measured'], 'vs30'] = \
        sta_df.loc[~sta_df['vs30measured'], 'VS30  ']
    sta_df['vs30'] = sta_df['VS30'].astype(float)

    metadata['vs30'] = np.nan
    metadata['vs30measured'] = False
    metadata['z1'] = np.nan
    metadata['z2pt5'] = np.nan

    for i, _ in sta_df.iterrows():
        flt = metadata['station_id'] == _['Site Code']
        if flt.any():  # noqa
            metadata.loc[flt, 'vs30'] = _['vs30']
            metadata.loc[flt, 'vs30measured'] = _['vs30measured']
            metadata.loc[flt, 'z1'] = _['ZP1.0']
            metadata.loc[flt, 'z2pt5'] = _['ZP2.5']

    metadata = metadata.set_index(["event_id", "station_id"], drop=False)
    return metadata


def find_sources(
    file_path: str,
    metadata: pd.DataFrame
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[pd.Series]]:
    """Find the file paths of the three waveform components, and their metadata

    :param file_path: the waveform path currently processed. it is one of the files
        accepted via `accept_file` and it should denote one of the three waveform
        components (the other two should be inferred from it)
    :param metadata: the Metadata dataframe. The returned waveforms metadata must be one
        row of this object as pandas Series (any other object will raise)

    :return: A tuple with three strings denoting the file absolute paths of the three
        components (horizontal1, horizontal2, vertical, in **this order** and the
        pandas Series denoting the waveforms metadata (common to the three components)
    """
    root, ext = splitext(file_path)
    if ext == '.EW':  # knet
        paths = file_path, root + ".NS", root + '.UD'
    elif ext == '.NS':
        paths = root + '.EW', file_path, root + '.UD'
    elif ext == '.UD':
        paths = root + '.EW', root + '.NS', file_path
    elif ext == '.EW1':
        paths = file_path, root + ".NS1", root + '.UD1'
    elif ext == '.NS1':
        paths = root + '.EW1', file_path, root + '.UD1'
    elif ext == '.UD1':
        paths = root + '.EW1', root + '.NS1', file_path
    elif ext == '.EW2':
        paths = file_path, root + ".NS2", root + '.UD2'
    elif ext == '.NS2':
        paths = root + '.EW2', file_path, root + '.UD2'
    elif ext == '.UD2':
        paths = root + '.EW2', root + '.NS2', file_path
    else:
        return None, None, None, None

    station_suffix = ''
    if file_path[-1:] in {'1', '2'}:
        station_suffix = f'.{file_path[-1:]}'

    record: Optional[pd.Series] = None
    ev_id = basename(dirname(file_path))
    sta_id = basename(root)[:6]  # station name is first 6 letters
    try:
        record = metadata.loc[(ev_id, sta_id)].copy()
        if not isinstance(record, pd.Series):  # multiple instances (safety check)
            raise KeyError()
        if station_suffix:
            sta_id = f'{sta_id}{station_suffix}'
        record["station_id"] = sta_id
        record["event_id"] = str(ev_id)
    except KeyError:
        pass

    return paths + (record, )


def read_waveform(file_path: str, content: BytesIO, metadata: pd.Series) -> Waveform:
    """Read a waveform from a file path

    :param file_path: the waveform path currently processed. It is one of the files
        accepted via `accept_file` and it should denote one of the three waveform
        components. You do not need to open the file here (see `content` parameter)
    :param content: a BytesIO (file-like) object with the content of file_path, as byte
        sequence
    :param metadata: the pandas Series related to the given file, as returned from
        `find_sources`

    :return: a `Waveform` object
    """
    scale_nom, scale_denom, dt = None, None, None
    for line in content:
        if line.startswith(b'Sampling Freq(Hz)'):
            dt = 1.0 / float(
                line.strip().lower().split(b' ')[-1].removesuffix(b'hz'))
        elif line.startswith(b'Scale Factor'):
            scale_str = line.split(b'  ', 1)[1].strip().split(b'/')
            scale_nom = float(scale_str[0][:-5])
            scale_denom = float(scale_str[1])
        elif line.startswith(b'Memo'):
            if any(_ is None for _ in (scale_nom, scale_denom, dt)):
                raise ValueError('dt /scale nom / scale denom not found')
            break
    rest = content.read()
    data = np.fromstring(rest, sep=" ", dtype=np.float32)
    # data = np.loadtxt(fp, dtype=np.float32)
    data *= scale_nom / scale_denom / 100.  # the 100. is to convert to m/s**2
    return Waveform(dt, data)


def post_process(
    metadata: pd.Series,
    h1: Optional[Waveform],
    h2: Optional[Waveform],
    v: Optional[Waveform]
) -> tuple[pd.Series, Optional[Waveform], Optional[Waveform], Optional[Waveform]]:
    """
    Custom post-processing on the metadata and waveforms read from disk.
    Typically, you complete metadata and waveforms, e.g. filling the former with missing
    fields, or converting the latter to the desired units (m/sec*sec, m/sec, m).
    **Remember** that Waveform objects are IMMUTABLE, so you need to return new
    Waveform object if modified

    :param metadata: the pandas Series related to the given file, as returned from
        `find_sources`. Non-standard fields do not need to be removed, missing standard
        fields will be filled with defaults (NaN, None or anything implemented in
        `metadata_fields.yml`)
    :param h1: the Waveform of the first horizontal component, or None (waveform N/A)
    :param h2: the Waveform of the second horizontal component, or None (waveform N/A)
    :param v: the Waveform of the vertical component, or None (waveform N/A)
    """
    orig_meta, metadata = metadata, metadata.copy()
    metadata['origin_time'] = datetime.fromisoformat(metadata["origin_time"])
    metadata['origin_time_resolution'] = 's'
    dt_format = "%Y%m%d%H%M%S"
    metadata["start_time"] = \
        datetime.strptime(str(metadata["start_time"]), dt_format)
    metadata["p_wave_arrival_time"] = \
        datetime.strptime(str(metadata["p_wave_arrival_time"]), dt_format)
    metadata["s_wave_arrival_time"] = \
        datetime.strptime(str(metadata["s_wave_arrival_time"]), dt_format)
    metadata['fault_type'] = {
        'S': 'Strike-Slip',
        'N': 'Normal',
        'R': 'Reverse'
    }.get(metadata['fault_type'])
    metadata["vs30measured"] = metadata["vs30measured"] in {1, "1", 1.0}
    metadata["region"] = 0
    metadata["filter_type"] = "A"
    metadata["filter_order"] = 4
    metadata["lower_cutoff_frequency_h2"] = metadata["lower_cutoff_frequency_h1"]
    metadata["upper_cutoff_frequency_h2"] = metadata["upper_cutoff_frequency_h1"]
    metadata["lowest_usable_frequency_h1"] = metadata["lower_cutoff_frequency_h1"]
    metadata["lowest_usable_frequency_h2"] = metadata["lower_cutoff_frequency_h2"]
    metadata['magnitude_type'] = {
        'J': 'MJ',  # JMA magnitude
        'D': 'MD',  # JMA displacement magnitude
        'd': 'Md',  # JMA displacement magnitude, but for two stations
        'V': 'MV',  # JMA velocity magnitude
        'v': 'Mv',  # JMA velocity magnitude, but for two or three stations
        'W': 'Mw',  # Moment magnitude
        'B': 'mb',  # Body wave magnitude from USGS
        'S': 'Ms',  # Surface wave magnitude from USGS
    }.get(metadata['magnitude_type'], None)
    return metadata, h1, h2, v


if __name__ == "__main__":
    import sys
    from common import main
    main(sys.modules[__name__])
