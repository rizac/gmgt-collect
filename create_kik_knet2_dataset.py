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


borehole_suffix = '_B'


def read_metadata(source_metadata_path: str, files: set[str]) -> pd.DataFrame:
    """
    Read and optionally pre-process the metadata Dataframe (e.g., setup index
    to easily find records from file names, optimize some column data like
    converting strings to categorical).

    :param source_metadata_path: the source metadata path (usually CSV)
    :param files: a set of file paths as returned from `scan_dir`

    :return: a pandas DataFrame optionally modified from `metadata`
    """

    # Mapping from source metadata columns to their new names.
    # Map to None to skip renaming and just load the column data
    source_metadata_fields = {
        'EQ_Code': 'event_id',
        "StationCode": 'station_id',

        'Origin_Meta': 'origin_time',
        'new_record_start_UTC': 'start_time',
        # 'RecordTime': None,
        #'tP_JMA': 'p_wave_arrival_time',
        # 'tS_JMA': 's_wave_arrival_time',
        # 'PGA_EW': None,
        # 'PGA_NS': None,
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
         "VS30": "vs30",
        # "vs30measured": "vs30measured",
        'StationLat.': "station_latitude",
        'StationLong.': "station_longitude",
        'StationHeight(m)': "station_height",
        'ZS1.5': "z1",
        # 'ZS2.5': "z2pt5",  # ADDED LATER (DEPENDS ON DATASET, see below)

        "fc0": "lower_cutoff_frequency_h1",  # FIXME CHECK THIS  hp_h1
        # "fc0": "lower_cutoff_frequency_h2",
        "fc1": "upper_cutoff_frequency_h1",
        # "fc1": "upper_cutoff_frequency_h2",
        # "fc0": "lowest_usable_frequency_h1",
        # "fc1": "lowest_usable_frequency_h2",  # if not sure, leave None
        'StationHeight(m)': 'station_elevation',
        # 'Zborehole': 'station_depth':   # ADDED ONLY FOR KIK (see below)
    }

    global borehole_suffix

    is_kik = splitext(next(iter(files)))[1][-1] in {'1', '2'}
    if is_kik:
        for borehole_key in ['Rhypo_B', 'RJB_0_B', 'RJB_1_B', 'Rrup_0_B',
                             'Rrup_1_B', 'fc0_B', 'fc1_B', 'PGA_rotd50_B']:
            normal_key = borehole_key[:-2]
            source_metadata_fields[borehole_key] = (
                source_metadata_fields[normal_key] + borehole_suffix
            )
        source_metadata_fields['Zborehole'] = 'station_depth'
        source_metadata_fields['ZS2.5'] = 'z2.5'
    else:
        source_metadata_fields['ZP2.5'] = 'z2.5'

    # These are the borehole columns (for ref):
    #
    # ['Address', 'EQ_Code', 'StationCode', 'Origin_Meta', 'RecordTime',
    # 'RecordTime_B', 'evLat._Meta', 'evLong._Meta', 'Depth. (km)_Meta',
    # 'Mag._Meta', 'NumberofStations', 'StationLat.', 'StationLong.',
    # 'StationHeight(m)', 'StationHeight(m)_B', 'Zborehole', 'Borehole_Processed',
    # 'samplingRate', 'samplingRate_B', 'JMA_match', 'JMA_Lat', 'JMA_Lon',
    # 'JMA_Depth', 'JMA_JST', 'JMA_UTC', 'JMA_Mag', 'JMA_Magtype', 'Fnet_match',
    # 'fnet_Latitude(deg)', 'fnet_Longitude(deg)', 'fnet_MT_Depth(km)',
    # 'fnet_JMA_Depth(km)', 'fnet_Number_of_Stations', 'fnet_MT_Magnitude(Mw)',
    # 'fnet_JMA_Magnitude(Mj)', 'fnet_Mo(Nm)', 'fnet_Origin_Time(UT)', 'fnet_Dip_0',
    # 'fnet_Dip_1', 'fnet_Rake_0', 'fnet_Rake_1', 'fnet_Strike_0', 'fnet_Strike_1',
    # 'fnet_Region_Name', 'fnet_Var._Red.', 'Focal_mechanism_BA', 'Repi', 'Rhypo',
    # 'Rhypo_B', 'RJB_0', 'RJB_0_B', 'RJB_1', 'RJB_1_B', 'Rrup_0', 'Rrup_0_B',
    # 'Rrup_1', 'Rrup_1_B', 'tP_JMA', 'tP_JMA_B', 'tP_STA_LTA', 'tP_STA_LTA_B',
    # 'tS_JMA', 'tS_JMA_B', 'new_record_start_UTC', 'new_record_start_UTC_B',
    # 'new_record_start_UTC_bool', 'new_record_start_UTC_bool_B', 'WaveType',
    # 'WaveType_B', 'length_record_s', 'length_record_s_B', 'length_raw_record_s',
    # 'length_raw_record_s_B', 'energy_ratioSignal', 'energy_ratioSignal_B',
    # 'MultFlag', 'MultFlag_B', 'duration_Noise', 'duration_Noise_B', 'noiseStart',
    # 'noiseStart_B', 'end_Swave', 'end_Swave_B', 'energy_ratioNoise',
    # 'energy_ratioNoise_B', 'fc0', 'fc0_B', 'fc1', 'fc1_B', 'freq_range',
    # 'freq_range_B', 'HighFreq_flag', 'HighFreq_flag_B', 'LowFreq_flag',
    # 'LowFreq_flag_B', 'snrEmean', 'snrEmean_B', 'snrNmean', 'snrNmean_B',
    # 'Dur5_75_E', 'Dur5_75_E_B', 'Dur5_75_N', 'Dur5_75_N_B', 'Dur5_95_E',
    # 'Dur5_95_E_B', 'Dur5_95_N', 'Dur5_95_N_B', 'AriasIntensity_E_B',
    # 'AriasIntensity_N_B', 'AriasIntensity_U_B', 'AriasIntensity_E',
    # 'AriasIntensity_N', 'AriasIntensity_U', 'CAV_E', 'CAV_E_B', 'CAV_N',
    # 'CAV_N_B', 'CAV_U', 'CAV_U_B', 'PGA_EW_Meta', 'PGA_EW_Meta_B', 'PGA_NS_Meta',
    # 'PGA_NS_Meta_B', 'PGA_EW', 'PGA_EW_B', 'PGA_NS', 'PGA_NS_B', 'PGV_EW',
    # 'PGV_EW_B', 'PGV_NS', 'PGV_NS_B', 'PGD_EW', 'PGD_EW_B', 'PGD_NS', 'PGD_NS_B',
    # 'PGA_rotd50', 'PGA_rotd50_B', 'PGV_rotd50', 'PGV_rotd50_B', 'PGD_rotd50',
    # 'PGD_rotd50_B', 'PGA_rotd0', 'PGA_rotd0_B', 'PGV_rotd0', 'PGV_rotd0_B',
    # 'PGD_rotd0', 'PGD_rotd0_B', 'PGA_rotd100', 'PGA_rotd100_B', 'PGV_rotd100',
    # 'PGV_rotd100_B', 'PGD_rotd100', 'PGD_rotd100_B', 'VS,surface', 'VS,borehole',
    # 'VS10', 'VS30', 'ZS1.5', 'ZS2.5', 'f0 HV']

    metadata = pd.read_csv(
        source_metadata_path, usecols=list(source_metadata_fields.keys())
    )
    metadata = metadata.rename(
        columns={k: v for k, v in source_metadata_fields.items() if v is not None}
    )

    # pre-process metadata dataframe (other station info):
    metadata = metadata.dropna(subset=['event_id', 'station_id'])
    metadata['event_id'] = metadata['event_id'].astype(str).astype('category')
    # add categorical according to stations ".1" or ".2" (borehole for kik):
    metadata['station_id'] = metadata['station_id'].astype(str)
    # duplicate stations and set proper columns for boreholes in case of kik-net:
    if is_kik:
        metadata_b = metadata.copy()
        rename_columns = {
            c + borehole_suffix: c for c in metadata_b.columns if c + borehole_suffix in metadata_b.columns
        }
        metadata_b.drop(columns=rename_columns.values(), inplace=True)
        metadata_b.rename(columns=rename_columns, inplace=True)
        metadata_b['station_id'] = metadata_b['station_id'].astype(str) + borehole_suffix
        metadata = pd.concat([metadata, metadata_b])
    else:
        metadata['station_depth'] = 0

    metadata['station_id'] = metadata['station_id'].astype('category')
    metadata['vs30measured'] = metadata['vs30'].notna()

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


    ev_id = basename(dirname(file_path))
    sta_id = basename(root)[:6]  # station name is first 6 letters
    if file_path[-1:] == '1':
        sta_id += borehole_suffix

    record: Optional[pd.Series] = None
    try:
        record = metadata.loc[(ev_id, sta_id)].copy()
        if not isinstance(record, pd.Series):  # multiple instances (safety check)
            record = None
            raise KeyError()
        record["station_id"] = sta_id
        record["event_id"] = str(ev_id)
    except KeyError:
        pass

    return paths[0], paths[1], paths[2], record


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
    # dt_format = "%Y%m%d%H%M%S"
    # metadata["start_time"] = \
    #     datetime.strptime(str(metadata["start_time"]), dt_format)
    # metadata["p_wave_arrival_time"] = \
    #     datetime.strptime(str(metadata["p_wave_arrival_time"]), dt_format)
    # metadata["s_wave_arrival_time"] = \
    #     datetime.strptime(str(metadata["s_wave_arrival_time"]), dt_format)
    metadata['fault_type'] = {
        'S': 'Strike-Slip',
        'N': 'Normal',
        'R': 'Reverse'
    }.get(metadata['fault_type'])
    # metadata["vs30measured"] = metadata["vs30measured"] in {1, "1", 1.0}
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
