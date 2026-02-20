"""
Python module to be executed as script to generate a new harmonized dataset from
heterogeneous sources. For a new dataset, copy rename this or any look-alike modules in
this directory and modify the editable part (see below). More details in README.md
"""
from __future__ import annotations

from typing import Optional
import os
from os.path import join, basename, dirname, splitext
import re
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
    b_name = basename(file_path)
    if b_name.startswith('RSN'):
        b_name, ext = splitext(b_name)
        if ext.startswith('.AT'):
            if b_name.endswith('UD') or b_name.endswith('DWN'):
                if b_name.find('_') > 3:
                    return True
    return False


# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {
    # 'header': None  # for CSVs with no header
    # 'dtype': {}  # NOT RECOMMENDED, see `metadata_fields.yml` instead
    # 'usecols': []  # NOT RECOMMENDED, see `source_metadata_fields` below instead
}

# Mapping from source metadata columns to their new names. Map to None to skip renaming
# and just load the column data
source_metadata_fields = {
    'EQID': "event_id",
    'Station ID  No.': None,
    "Station Sequence Number": None,

    # "fpath_h1": None,
    # "fpath_h2": None,
    # "fpath_v": None,
    'Record Sequence Number': None,

    "EpiD (km)": "epicentral_distance",
    "HypD (km)": "hypocentral_distance",
    "Joyner-Boore Dist. (km)": "joyner_boore_distance",
    "ClstD (km)": "rupture_distance",
    "Rx": "fault_normal_distance",
    'YEAR': None,
    'MODY': None,
    'HRMN': None,
    "Hypocenter Latitude (deg)": "event_latitude",
    "Hypocenter Longitude (deg)": "event_longitude",
    "Hypocenter Depth (km)": "event_depth",
    "Earthquake Magnitude": "magnitude",
    "Magnitude Type": "magnitude_type",
    "Depth to Top Of Fault Rupture Model": "depth_to_top_of_fault_rupture",
    "Fault Rupture Width (km)": "fault_rupture_width",
    "Strike (deg)": "strike",
    "Dip (deg)": "dip",
    "Rake Angle (deg)": "rake",

    "Mechanism Based on Rake Angle": "fault_type",
    "Vs30 (m/s) selected for analysis": "vs30",
    # vs30measured is a boolean expression; treated as key
    "Measured/Inferred Class": "vs30measured",
    "Station Latitude": "station_latitude",
    "Station Longitude": "station_longitude",
    "Northern CA/Southern CA - H11 Z1 (m)": "z1",
    "Northern CA/Southern CA - H11 Z2.5 (m)": "z2pt5",

    "Type of Filter": "filter_type",
    "npass": None,
    "nroll": None,
    "HP-H1 (Hz)": "lower_cutoff_frequency_h1",
    "HP-H2 (Hz)": "lower_cutoff_frequency_h2",
    "LP-H1 (Hz)": "upper_cutoff_frequency_h1",
    "LP-H2 (Hz)": "upper_cutoff_frequency_h2",
    "Lowest Usable Freq - H1 (Hz)": "lowest_usable_frequency_h1",
    "Lowest Usable Freq - H2 (H2)": "lowest_usable_frequency_h2",

    "PGA (g)": "PGA"
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
    # set event and station categorical (save space)
    no_sta_id = metadata['Station ID  No.'].str.startswith("-999")
    metadata['station_id'] = metadata['Station ID  No.']
    metadata.loc[no_sta_id, 'station_id'] = \
        "SSN_" + metadata.loc[no_sta_id, 'Station Sequence Number'].astype(str)
    metadata.drop(columns=['Station ID  No.', 'Station Sequence Number'], inplace=True)
    metadata.dropna(subset=['event_id', 'station_id'], inplace=True)
    metadata['event_id'] = metadata['event_id'].astype(str).astype('category')
    metadata['station_id'] = metadata['station_id'].astype('category')
    metadata['Record Sequence Number'] = metadata['Record Sequence Number'].astype(int)
    assert not metadata['Record Sequence Number'].duplicated().any()
    metadata.set_index(['Record Sequence Number'], drop=True, inplace=True)
    return metadata


def find_sources(
    file_path: str, metadata: pd.DataFrame
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
    file_basename = basename(file_path)
    b_name, ext = splitext(file_basename)
    root_dir = dirname(file_path)
    rsn = None
    try:
        rsn = int(file_basename[3:].split('_')[0])
    except (ValueError, TypeError):
        pass
    if rsn is not None:
        try:
            meta = metadata.loc[rsn]  # connot return a Series (slices in loc)
            if isinstance(meta, pd.Series):

                if b_name.endswith('DWN'):
                    ptrn = re.compile(re.escape(b_name[:-3]) + r"\d+" + re.escape(ext))
                    # search in the ame directory:
                    filez = []
                    for fle in os.listdir(dirname(file_path)):
                        if fle != file_basename:
                            if ptrn.match(fle):
                                filez.append(join(root_dir, fle))
                    if len(filez) == 0:
                        filez = [None, None]
                    elif len(filez) == 1:
                        filez.append(None)
                    elif len(filez) != 2:
                        raise KeyError()  # see below
                    filez.append(file_path)
                else:  # UD
                    filez = (
                        join(root_dir, b_name[:-2] + 'NS' + ext),
                        join(root_dir, b_name[:-2] + 'EW' + ext),
                        file_path
                    )
                return tuple(filez) + (meta,)  # convert to Series

        except KeyError:
            pass

    return None, None, None, None


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

    # First few lines are headers
    header1 = content.readline().strip()
    header2 = content.readline().strip()
    header3 = content.readline().strip()
    header4 = content.readline().split(b",")
    npts = int(re.match(br"NPTS\s*=\s*(\d+)", header4[0].strip()).group(1))
    dt = float(re.match(br"DT\s*=\s*([\.\d]+)\s*SEC", header4[1].strip()).group(1))
    data_str = b" ".join(line for line in content)
    # The acceleration time series is given in units of g. So I convert it in m/s.
    return Waveform(dt, np.fromstring(data_str, sep=" ") * 9.80665)


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
    # convert time(s):
    year = metadata['YEAR']
    month_day = str(metadata['MODY'])
    if month_day in (-999, '-999') or pd.isna(month_day):
        raise AssertionError('Invalid month_day')
    month_day = month_day.zfill(4)  # pad with zeroes
    month, day = int(month_day[:2]), int(month_day[2:])

    hour_min = str(metadata['HRMN'])
    if hour_min in (-999, '-999') or pd.isna(hour_min):
        evt_time = datetime(
            year=year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0
        )
        ot_res = 'D'
    else:
        hour_min = hour_min.zfill(4)  # pad with zeroes
        hour, mins = int(hour_min[:2]), int(hour_min[2:])
        evt_time = datetime(
            year=year, month=month, day=day, hour=hour, minute=mins, second=0,
            microsecond=0
        )
        ot_res = 'm'
    # use datetimes also for event_date (for simplicity when casting later):
    metadata["origin_time"] = evt_time
    metadata["origin_time_resolution"] = ot_res

    metadata['filter_order'] = 0  # the default (unknown)
    if metadata["filter_type"] in (-999, '-999'):
        metadata["filter_type"] = None
    elif metadata['filter_type'] == 'A':
        # this link: https://peer.berkeley.edu/sites/default/files/webpeer-2014-20-mayssa_dabaghi_armen_der_kiureghian.pdf  # noqa
        # states that they used a 5-th order acasual butterwoirth. Most of the data
        # has nroll=2.5 and npass =1. I assume that for these cases, if filter_type = A,
        # then the order is 5 (consistent with nroll and npass):
        if metadata['nroll'] in {2.5, '2.5'} and metadata['npass'] in {1, '1'}:
            metadata['filter_order'] = 5

    if metadata['magnitude_type'] == 'U':
        metadata['magnitude_type'] = None

    try:
        metadata['fault_type'] = [
            'Strike-Slip', 'Normal', 'Reverse', 'Reverse-Oblique', 'Normal-Oblique'
        ][int(metadata['fault_type'])]
    except (IndexError, ValueError, TypeError):
        metadata['fault_type'] = None

    # convert from g to m/s2:
    metadata["PGA"] = metadata["PGA"] * 9.80665

    # vs30 measured has weird values 2a_3a_3b ore whatever. Set to inferred in this case
    metadata['vs30measured'] = metadata['vs30measured'] in {0, '0'}

    # simply return the arguments (no processing by default):
    return metadata, h1, h2, v


if __name__ == "__main__":
    import sys
    from common import main
    main(sys.modules[__name__])
