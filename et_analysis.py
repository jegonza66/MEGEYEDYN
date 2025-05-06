import os
import pyxations as pyx
import paths
import re


def parse_asc_file(asc_file_path):
    # Initialize lists for onsets and offsets
    onsets = []
    offsets = []

    # Lists to store timestamps of relevant messages
    ttl_80_timestamps = []  # Trial start (TTL 80)
    ttl_64_timestamps = []  # Pause (TTL 64)
    ttl_96_timestamps = []  # Block end (TTL 96)

    # Regular expression to match MSG lines with TTL_Signal_Value
    msg_pattern = r'MSG\s+(\d+)\s+-?\d+\s+!V TRIAL_VAR TTL_Signal_Value\s+(\d+)'

    # Read and parse the ASC file
    with open(asc_file_path, 'r') as file:
        for line in file:
            match = re.match(msg_pattern, line.strip())
            if match:
                timestamp, ttl_value = map(int, match.groups())
                if ttl_value == 80:
                    ttl_80_timestamps.append(timestamp)
                elif ttl_value == 64:
                    ttl_64_timestamps.append(timestamp)
                elif ttl_value == 96:
                    ttl_96_timestamps.append(timestamp)

    # Process trials (expecting 33 trials: 3 blocks of 11 trials)
    trial_count = 0
    block_count = 0
    ttl_64_index = 1

    for i, onset in enumerate(ttl_80_timestamps):
        trial_count += 1
        onsets.append(onset)

        # Determine if this is the last trial of a block (11th trial)
        if trial_count % 11 == 0:
            # Use TTL 96 for the block end (match by block count)
            if block_count < len(ttl_96_timestamps):
                offsets.append(ttl_96_timestamps[block_count])
                block_count += 1
                ttl_64_index += 1
        else:
            # Use the next TTL 64 as the offset
            if ttl_64_index < len(ttl_64_timestamps):
                offsets.append(ttl_64_timestamps[ttl_64_index])
                ttl_64_index += 1

        # Reset trial count after each block
        if trial_count == 11:
            trial_count = 0

    return onsets, offsets


edfs_path = paths.et_path + 'edfs/'
bids_path = paths.main_path
bids_name = 'DATA_bids_final'

timestamps_onset = {}
timestamps_offset = {}

for edf_file in os.listdir(edfs_path):
    subject_id = edf_file.split('_1')[0]
    asc_file_path = os.path.join(paths.et_path, subject_id, subject_id + '.asc')
    onset_subject, offset_subject = parse_asc_file(asc_file_path)
    timestamps_onset[subject_id] = {'1': {'video': onset_subject}}
    timestamps_offset[subject_id] = {'1': {'video': offset_subject}}

bids_dataset_folder = pyx.dataset_to_bids(target_folder_path=bids_path, files_folder_path=edfs_path, dataset_name=bids_name)

pyx.compute_derivatives_for_dataset(bids_dataset_folder=bids_dataset_folder, dataset_format="eyelink", detection_algorithm='eyelink',
                                    msg_keywords=['TTL_Signal_Value 80', 'TTL_Signal_Value 64', 'TTL_Signal_Value 96', 'Source_File cgi_video', 'TRIAL_VAR Start_Frame'],
                                    start_times=timestamps_onset, end_times=timestamps_offset)


##
import pyxations as pyx
import paths
import pandas as pd
from pyxations import Experiment

edfs_path = paths.et_path + 'edfs/'
bids_path = paths.main_path
bids_name = 'DATA_bids'

exp = Experiment(rf"{bids_path}{bids_name}")
exp.load_data("eyelink")

exp.drop_trials_with_nan_threshold("search",0.1)
exp.collapse_fixations(80.0)
exp.filter_fixations()
exp.plot_multipanel(True)
exp.plot_scanpaths(screen_height=1080, screen_width=1920)

exp.plot_calib_data()