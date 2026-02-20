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
    return splitext(file_path)[1].startswith('.ASC')


# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {
    'sep': ';'
    # 'header': None  # for CSVs with no header
    # 'dtype': {}  # NOT RECOMMENDED, see `metadata_fields.yml` instead
    # 'usecols': []  # NOT RECOMMENDED, see `source_metadata_fields` below instead
}

# Mapping from source metadata columns to their new names. Map to None to skip renaming
# and just load the column data
source_metadata_fields = {
    'event_id': "event_id",
    "network_code" :None,
    "station_code": None,
    "location_code": None,
    "instrument_code": None,

    "epi_dist": "epicentral_distance",
    # "?": "hypocentral_distance",
    "JB_dist": "joyner_boore_distance",
    "rup_dist": "rupture_distance",
    "Rx_dist": "fault_normal_distance",
    'event_time': "origin_time",

    "ev_latitude": "event_latitude",
    "ev_longitude": "event_longitude",
    "ev_depth_km": "event_depth",
    "EMEC_Mw": "magnitude",
    "Mw": None,
    "ML": None,
    "Ms": None,
    "es_z_top": "depth_to_top_of_fault_rupture",
    "es_width": "fault_rupture_width",
    "es_strike": "strike",
    "es_dip": "dip",
    "es_rake": "rake",

    "fm_type_code": "fault_type",
    "vs30_m_sec": "vs30",
    'vs30_meas_type': None,
    'vs30_m_sec_WA': None,

    # vs30measured is a boolean expression; treated as key
    # "Measured/Inferred Class": "vs30measured",
    "st_latitude": "station_latitude",
    "st_longitude": "station_longitude",
    # "Northern CA/Southern CA - H11 Z1 (m)": "z1",
    # "Northern CA/Southern CA - H11 Z2.5 (m)": "z2pt5",

    'U_channel_code': None,
    'W_channel_code': None,
    'V_channel_code': None,
    'U_hp': None,
    'V_hp': None,
    'W_hp': None,
    'U_lp': None,
    'V_lp': None,
    'W_lp': None,

    'rotD50_pga': 'PGA',
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
    cols = ["network_code", "station_code", "location_code", "instrument_code"]
    for c in cols:
        metadata[c] = metadata[c].astype(str)
    metadata = metadata.dropna(subset=cols + ['event_id'])
    metadata['station_id'] = metadata[cols].agg('.'.join, axis=1)
    metadata = metadata.drop(columns=cols)

    # set the categories for station_id:
    new_station_ids = {".".join(basename(f).split('.')[:4])[:-1] for f in files}
    station_ids = set(metadata['station_id'])
    assert new_station_ids & station_ids
    station_ids.update(new_station_ids)
    metadata['station_id'] = metadata['station_id'].astype(
        pd.CategoricalDtype(categories=station_ids, ordered=True)
    )

    new_event_ids = {basename(dirname(f)).removesuffix(".zip") for f in files}
    event_ids = set(metadata['event_id'])
    assert event_ids & event_ids
    event_ids.update(new_event_ids)
    metadata['event_id'] = metadata['event_id'].astype(str).astype(
        pd.CategoricalDtype(categories=event_ids, ordered=True)
    )

    metadata['magnitude_type'] = 'Mw'
    mag_missing = metadata['magnitude'].isna()
    metadata.loc[mag_missing, 'magnitude_type'] = None
    cols = ['Mw', 'Ms', 'ML']
    for mag_type in cols:
        mag_to_be_set = mag_missing & metadata[mag_type].notna()
        if mag_to_be_set.any():
            metadata.loc[mag_to_be_set, 'magnitude'] = metadata[mag_type][mag_to_be_set]
            metadata.loc[mag_to_be_set, 'magnitude_type'] = mag_type
            mag_missing = mag_missing & (~mag_to_be_set)
    metadata = metadata.drop(columns=cols)

    metadata['origin_time'] = pd.to_datetime(metadata['origin_time'])
    metadata['origin_time_resolution'] = 's'

    fault_types = {
        'SS': 'Strike-Slip',
        'NF': 'Normal',
        'TF': 'Reverse',
        'O': 'Normal-Oblique'
    }
    metadata.loc[~metadata['fault_type'].isin(fault_types.keys()), 'fault_type'] = None
    for key, repl in fault_types.items():
        metadata.loc[metadata['fault_type'] == key, 'fault_type'] = repl

    metadata['vs30measured'] = ~pd.isna(metadata.pop('vs30_meas_type'))
    metadata.loc[pd.notna(metadata['vs30']), 'vs30measured'] = True
    vs30_wa = metadata.pop('vs30_m_sec_WA')
    set_vs30 = pd.isna(metadata['vs30']) & pd.notna(vs30_wa)
    metadata.loc[set_vs30, 'vs30'] = vs30_wa[set_vs30]
    metadata.loc[set_vs30, 'vs30measured'] = False

    metadata["lower_cutoff_frequency_h1"] = np.nan
    metadata["lowest_usable_frequency_h1"] = np.nan
    metadata["lower_cutoff_frequency_h2"] = np.nan
    metadata["lowest_usable_frequency_h2"] = np.nan
    metadata["upper_cutoff_frequency_h1"] = np.nan
    metadata["upper_cutoff_frequency_h2"] = np.nan

    cols = {
        'U_channel_code': ('U_hp', 'U_lp'),
        'V_channel_code': ('V_hp', 'V_lp'),
        'W_channel_code': ('W_hp', 'W_lp')
    }
    for ch_code_col, (hp_col, lp_col) in cols.items():
        hp_values = metadata[hp_col]
        lp_values = metadata[lp_col]

        north_south = metadata[ch_code_col] == 'N'
        metadata.loc[north_south, "lower_cutoff_frequency_h1"] = hp_values[north_south]
        metadata.loc[north_south, "higher_cutoff_frequency_h1"] = lp_values[north_south]
        metadata.loc[north_south, "lowest_usable_frequency_h1"] = hp_values[north_south]

        east_west = metadata[ch_code_col] == 'E'
        metadata.loc[east_west, "lower_cutoff_frequency_h2"] = hp_values[east_west]
        metadata.loc[east_west, "higher_cutoff_frequency_h2"] = lp_values[east_west]
        metadata.loc[east_west, "lowest_usable_frequency_h2"] = hp_values[east_west]

        metadata = metadata.drop(columns=[ch_code_col, hp_col, lp_col])

    metadata['PGA'] = abs(metadata['PGA']) / 100  # from cm/sec2 to m/sec2

    metadata = metadata.set_index(['event_id', 'station_id'], drop=False)
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
    ev_id = splitext(basename(dirname(file_path)))[0]
    sta_id = ".".join(basename(file_path).split('.')[:4])
    file_suffix = basename(file_path).removeprefix(sta_id)
    orientation = sta_id[-1]
    sta_id = sta_id[:-1]
    if orientation in {'N', 'E', 'Z'}:  # we ignore  '1', '2', '3' for the moment
        try:
            meta = metadata.loc[(ev_id, sta_id)]
        except KeyError:
            meta = pd.Series()

        file_path_n = join(dirname(file_path), f'{sta_id}N{file_suffix}')
        file_path_e = join(dirname(file_path), f'{sta_id}E{file_suffix}')
        file_path_z = join(dirname(file_path), f'{sta_id}Z{file_suffix}')

        if orientation == 'N':
            return file_path, file_path_e, file_path_z, meta
        elif orientation == 'E':
            return file_path_n, file_path, file_path_z, meta
        else:
            return file_path_n, file_path_e, file_path, meta

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
    if 'SL.ILBA..HGZ.D.EMSC-20140313_0000055.ACC.MP.ASC' in file_path:
        asd = 9
    factor = None
    dt = None
    pos = 0
    metadata_tmp = {}
    comp_idx = None
    for line in content:
        pos = content.tell()  # remember current position
        line = line.strip()
        if b':' not in line:
            break
        line = line.decode('utf8')
        key, val = line.split(':', 1)
        key, val = key.strip(), val.strip()
        if key == 'DATA_TYPE':
            assert val == 'ACCELERATION', f'Invalid data type: {val}'
        elif key == 'UNITS':
            assert val in ('cm/s^2', 'm/s^2', 'g'), f'Invalid unit: {val}'
            if val == 'cm/s^2':
                factor = 0.01
            elif val == 'g':
                factor = 9.80665
        elif key == 'SAMPLING_INTERVAL_S':
            dt = float(val)
        elif key == 'STREAM':
            comp_idx = {'N': 0, 'E': 1, 'Z': 2}[val[2]]

        if ('DATA_CITATION' in key or 'DATA_CREATOR' in key or 'DATA_MEDIATOR' in key or
                '_REFERENCE' in key):
            continue
        metadata_tmp[key] = val

    for key, val in metadata_tmp.items():
        # next line is kinds dict.setdefault:
        metadata.loc[f'.{key}'] = metadata.get(f'.{key}', [None, None, None])
        metadata.loc[f'.{key}'][comp_idx] = val

    assert dt is not None, 'dt not found in file'

    content.seek(pos)
    # Load data into numpy array from that line onward
    data = np.fromstring(content.read().decode('utf-8'), sep='\n', dtype=float)
    if factor is not None:
        data *= factor

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
    # metadata also contains *most* of the fields below (read from each waveform file,
    # see `read_waveform`) prefixed with a dot to avoid conflicts. Each field is mapped
    # to a list of strings or None corresponding to the [h1, h2, v] components.

    # EVENT_NAME: None
    # EVENT_ID: TK-2000-0449
    # EVENT_DATE_YYYYMMDD: 20000823
    # EVENT_TIME_HHMMSS: 134126
    # EVENT_LATITUDE_DEGREE: 40.7820
    # EVENT_LONGITUDE_DEGREE: 30.7600
    # EVENT_DEPTH_KM: 10.5
    # HYPOCENTER_REFERENCE: ISC-webservice
    # MAGNITUDE_W: 5.2
    # MAGNITUDE_W_REFERENCE: Pondrelli_et_al_2002_dataset
    # MAGNITUDE_L:
    # MAGNITUDE_L_REFERENCE:
    # FOCAL_MECHANISM: Strike-slip faulting
    # NETWORK: TK
    # STATION_CODE: 1001
    # STATION_NAME: AI_146_BLK
    # STATION_LATITUDE_DEGREE: 39.650030
    # STATION_LONGITUDE_DEGREE: 27.856860
    # STATION_ELEVATION_M:
    # LOCATION: 00
    # SENSOR_DEPTH_M:
    # VS30_M/S:
    # SITE_CLASSIFICATION_EC8: B (inferred from topography)
    # MORPHOLOGIC_CLASSIFICATION:
    # EPICENTRAL_DISTANCE_KM: 277.2
    # EARTHQUAKE_BACKAZIMUTH_DEGREE: 244.0
    # DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS: 20000823_134250.000
    # DATE_TIME_FIRST_SAMPLE_PRECISION: seconds
    # SAMPLING_INTERVAL_S: 0.005000
    # NDATA: 18904
    # DURATION_S: 94.520
    # STREAM: HNE
    # UNITS: cm/s^2
    # INSTRUMENT: HN
    # INSTRUMENT_ANALOG/DIGITAL: D
    # INSTRUMENTAL_FREQUENCY_HZ:
    # INSTRUMENTAL_DAMPING:
    # FULL_SCALE_G:
    # N_BIT_DIGITAL_CONVERTER:
    # PGA_CM/S^2: 1.326285
    # TIME_PGA_S: 30.350000
    # BASELINE_CORRECTION: BASELINE REMOVED
    # FILTER_TYPE: BUTTERWORTH
    # FILTER_ORDER: 2
    # LOW_CUT_FREQUENCY_HZ: 0.200
    # HIGH_CUT_FREQUENCY_HZ: 20.000
    # LATE/NORMAL_TRIGGERED: NT
    # DATABASE_VERSION: 0.5
    # HEADER_FORMAT: DYNA 1.2
    # DATA_TYPE: ACCELERATION
    # PROCESSING: manual (Paolucci et al., 2011)
    # DATA_TIMESTAMP_YYYYMMDD_HHMMSS: 20250806_133936.520
    # DATA_LICENSE: U (unknown license)
    # DATA_CITATION: <NOT PRESENT (see above)>
    # DATA_CREATOR: <NOT PRESENT (see above)>
    # ORIGINAL_DATA_MEDIATOR_CITATION: <NOT PRESENT (see above)>
    # ORIGINAL_DATA_MEDIATOR: <NOT PRESENT (see above)>
    # ORIGINAL_DATA_CREATOR_CITATION: <NOT PRESENT (see above)>
    # ORIGINAL_DATA_CREATOR: network: <NOT PRESENT (see above)>

    not_na = pd.notna
    is_na = pd.isna

    # first extract the stuff that is component dependent:
    file_cutoff_freqs = {}
    try:
        freq_h1, freq_h2, _ = np.array(metadata.get(
            '.LOW_CUT_FREQUENCY_HZ', [np.nan, np.nan, np.nan]
        ), dtype=float)
    except (ValueError, TypeError, IndexError):
        freq_h1, freq_h2 = np.nan, np.nan
    file_cutoff_freqs['lower_cutoff_frequency_h1'] = freq_h1
    file_cutoff_freqs['lowest_usable_frequency_h1'] = freq_h1
    file_cutoff_freqs['lower_cutoff_frequency_h2'] = freq_h2
    file_cutoff_freqs['lowest_usable_frequency_h2'] = freq_h2
    try:
        freq_h1, freq_h2, _ = np.array(metadata.get(
            '.HIGH_CUT_FREQUENCY_HZ', [np.nan, np.nan, np.nan]
        ), dtype =float)
    except (ValueError, TypeError, IndexError):
        freq_h1, freq_h2 = np.nan, np.nan
    file_cutoff_freqs['upper_cutoff_frequency_h1'] = freq_h1
    file_cutoff_freqs['upper_cutoff_frequency_h2'] = freq_h2

    try:
        pga1, pga2, pga3 = np.array(metadata.get(
            '.PGA_CM/S^2', [np.nan, np.nan, np.nan]
        ), dtype=float)
    except (ValueError, TypeError, IndexError):
        pga1, pga2, pga3 = np.nan, np.nan, np.nan
    if not_na(pga1) and not_na(pga2):
        pga = np.sqrt(np.abs(pga1) * np.abs(pga2)) / 100
    elif not_na(pga3):
        pga = np.abs(pga3) / 100
    else:
        pga = np.nan
    # first thing: prune file metadata that are None because we do not have the
    # corresponding component. So if metadata['SOME_KEY'] is [None, None, 'a'] and
    # we do not have horizontal components, set it to ['a']. This is needed cause we
    # can then check the uniqueness of ['a'] here below
    nones = [h1 is None, h2 is None, v is None]
    if any(nones):
        for key in metadata.keys():
            if key.startswith('.'):
                vals = metadata[key]
                if isinstance(vals, list):
                    new_vals = [v for v, rm in zip(vals, nones) if not rm]
                    metadata[key] = new_vals

    if is_na(metadata.get('event_id')):
        assert len(set(metadata.get(".EVENT_ID", []))) == 1, 'No unique event id in file'
        metadata['event_id'] = metadata[".EVENT_ID"][0]

    # process remaining data:
    if is_na(metadata.get('station_id')):
        new_id = []
        for key in ['.NETWORK', '.STATION_CODE', '.LOCATION', '.INSTRUMENT']:
            vals = metadata.get(key, [])
            assert len(set(vals)) == 1, f'No unique {key} in file'
            new_id.append(vals[0])
        metadata['station_id'] = ".".join(new_id)

    if 'filter_type' not in metadata:  # noqa
        file_filt_types = metadata.get('.FILTER_TYPE', [])
        if len(set(file_filt_types)) == 1:
            file_filt_type = file_filt_types[0]
            if file_filt_type == 'BUTTERWORTH':
                metadata['filter_type'] = 'A'
                filter_order = 0
                filter_orders = metadata.get('.FILTER_ORDER', [])
                if len(set(filter_orders)) == 1:
                    filter_order = int(filter_orders[0])
                metadata['filter_order'] = filter_order

    for key, val in file_cutoff_freqs.items():
        if is_na(metadata.get(key)) and not_na(val):
            metadata[key] = val

    if is_na(metadata.get('magnitude')):
        for mag_type, file_mag_label in [('Mw', '.MAGNITUDE_W'), ('ML', '.MAGNITUDE_L')]:
            file_mags = metadata.get(file_mag_label, [])
            if len(set(file_mags)) == 1:
                file_mag = file_mags[0]
                if not_na(file_mag):
                    try:
                        metadata['magnitude'] = float(file_mag)
                        metadata['magnitude_type'] = mag_type
                        break
                    except ValueError:
                        pass

    if is_na(metadata.get('fault_type')):
        file_foc_mecs = metadata.get('.FOCAL_MECHANISM', [])
        if len(set(file_foc_mecs)) == 1:
            file_foc_mec = file_foc_mecs[0]
            file_foc_mec = file_foc_mec.strip().removesuffix(' faulting').strip().lower()
            if file_foc_mec != 'unknown':
                if file_foc_mec == 'thrust':
                    file_foc_mec = 'reverse'
                # check lower case and other stuff:
                for def_foc_mec in [
                    'Strike-Slip',
                    'Normal',
                    'Reverse',
                    'Reverse-Oblique',
                    'Normal-Oblique'
                ]:
                    if file_foc_mec == def_foc_mec.lower():
                        metadata['fault_type'] = def_foc_mec

    if is_na(metadata.get('start_time')):
        file_st_times = metadata.get('.DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS', [])
        if len(set(file_st_times)) == 1:
            file_st_time = file_st_times[0]
            if not_na(file_st_time) and len(file_st_time.strip()):
                for fmt in ("%Y%m%d_%H%M%S.%f", "%Y%m%d_%H%M%S"):
                    try:
                        metadata['start_time'] = datetime.strptime(file_st_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(
                        f"Cannot parse start_time from waveform file: "
                        f"{metadata['.DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS']}"
                    )

    if is_na(metadata.get('PGA')) and not_na(pga):
        metadata['PGA'] = pga

    if is_na(metadata.get('origin_time')):
        event_dates = metadata.get('.EVENT_DATE_YYYYMMDD', [])
        if len(set(event_dates)) == 1:
            event_date = event_dates[0]
            year = int(event_date[:4])
            month = int(event_date[4:6])
            day = int(event_date[6:])
            hour, minute, second = 0, 0, 0
            metadata['origin_time_resolution'] = 'D'
            event_times = metadata.get('.EVENT_TIME_HHMMSS', [])
            if len(set(event_times)) == 1:
                event_time = event_times[0]
                hour = int(event_time[:2])
                minute = int(event_time[2:4])
                second = int(event_time[4:6])
                metadata['origin_time_resolution'] = 's'
            metadata['origin_time'] = datetime(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=0
            )

    if is_na(metadata.get('start_time')):
        s_times = metadata.get(".DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS", [])
        if len(set(s_times)) == 1:
            metadata['start_time'] = datetime.strptime( s_times[0], "%Y%m%d_%H%M%S")

    # set float elements:
    for key, new_key in {
        # 'EVENT_ID': 'event_id',
        'EVENT_LATITUDE_DEGREE': 'event_latitude',
        'EVENT_LONGITUDE_DEGREE': 'event_longitude',
        'EVENT_DEPTH_KM': 'event_depth',
        'STATION_LATITUDE_DEGREE': 'station_latitude',
        'STATION_LONGITUDE_DEGREE': 'station_longitude',
        # 'STATION_ELEVATION_M': 'station_height',
        # 'SENSOR_DEPTH_M': None,  # FIXME CHECK
        'VS30_M/S': 'vs30',
        # 'SITE_CLASSIFICATION_EC8': None,
        'EPICENTRAL_DISTANCE_KM': 'epicentral_distance',
        # 'DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS': 'start_time',
    }.items():
        if is_na(metadata.get(new_key)):
            values = metadata.get(f".{key}", [])
            if len(set(values)) == 1:
                try:
                    metadata[new_key] = float(values[0])
                except ValueError:
                    pass

    if is_na(metadata.get('vs30')):
        file_ec8_classes = metadata.get('.SITE_CLASSIFICATION_EC8', [])
        if len(set(file_ec8_classes)) == 1:
            file_ec8_class = file_ec8_classes[0]
            idx2remove = file_ec8_class.find(' (inferred ')
            if idx2remove > -1:
                file_ec8_class = file_ec8_class[:idx2remove]
            val = {
                "A": 900,
                "B": 580,
                "C": 270,
                "D": 150,
                "E": 100
            }.get(file_ec8_class)
            if not_na(val):
                metadata['vs30'] = val
                metadata['vs30measured'] = False

    if not_na(metadata.get('epicentral_distance')) and \
            not_na(metadata.get('event_depth')) and \
            is_na(metadata.get('hypocentral_distance')):
        metadata['hypocentral_distance'] = np.sqrt(
            (metadata['epicentral_distance'] ** 2) + metadata['event_depth'] ** 2
        )

    return metadata, h1, h2, v


if __name__ == "__main__":
    import sys
    from common import main
    main(sys.modules[__name__])
