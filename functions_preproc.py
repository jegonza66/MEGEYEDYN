import copy
import numpy as np
import pandas as pd
import os
import math
import mne
import scipy.signal as sgn
import functions_general
import plot_preproc
import paths
import subprocess
import shutil


def convert_edf_to_ascii(edf_file_path, output_dir):
    """
    Convert an EDF file to ASCII format using edf2asc.

    Args:
        edf_file_path (str): Path to the input EDF file.
        output_dir (str): Directory to save the ASCII file. If None, the ASCII file will be saved in the same directory as the input EDF file.

    Returns:
        str: Path to the generated ASCII file.
    """
    # Check if edf2asc is installed
    if not shutil.which("edf2asc"):
        raise FileNotFoundError("edf2asc not found. Please make sure EyeLink software is installed and accessible in the system PATH.")

    # Set output directory
    if output_dir is None:
        raise ValueError("Output directory must be specified.")

    # Generate output file path
    edf_file_name = os.path.basename(edf_file_path)
    ascii_file_name = os.path.splitext(edf_file_name)[0] + ".asc"
    ascii_file_path = os.path.join(output_dir, ascii_file_name)

    # Run edf2asc command with the -f flag, only run it if the file does not already exist
    if not os.path.exists(ascii_file_path):
        subprocess.run(["edf2asc", "-f", edf_file_path, ascii_file_path])

    return ascii_file_path


def reescale_et_channels(meg_gazex_data_raw, meg_gazey_data_raw, minvoltage=-5, maxvoltage=5, minrange=-0.2, maxrange=1.2,
                         screenright=1919, screenleft=0, screentop=0, screenbottom=1079):
    """
    Reescale Gaze data from MEG Eye Tracker channels to correct digital to analog conversion.

    Parameters
    ----------
    meg_gazex_data_raw: ndarray
        Raw gaze x data to rescale
    meg_gazey_data_raw: ndarray
        Raw gaze x data to rescale
    minvoltage: int
        The minimum voltage value from the digital-analog conversion.
        Default to -5 from analog.ini analog_dac_range.
    maxvoltage: int
        The maximum voltage value from the digital-analog conversion.
        Default to 5 from analog.ini analog_dac_range.
    minrange: int
        The minimum gaze position tracked by the eye tracker outside the screen.
        Default to -0.2 from analog.ini analog_x_range to allow for +/- 20% outside display
    maxrange: int
        The maximum gaze position tracked by the eye tracker outside the screen.
        Default to 1.2 from analog.ini analog_x_range to allow for +/- 20% outside display
    screenright: int
        Pixel number of the right side of the screen. Default to 1919.
    screenleft: int
        Pixel number of the left side of the screen. Default to 0.
    screentop: int
        Pixel number of the top side of the screen. Default to 0.
    screenbottom: int
        Pixel number of the bottom side of the screen. Default to 1079.

    Returns
    -------
    meg_gazex_data_scaled: ndarray
        The scaled gaze x data
    meg_gazey_data_scaled: ndarray
        The scaled gaze y data
    """

    print('Rescaling')
    # Scale
    R_h = (meg_gazex_data_raw - minvoltage) / (maxvoltage - minvoltage)  # voltage range proportion
    S_h = R_h * (maxrange - minrange) + minrange  # proportion of screen width or height
    R_v = (meg_gazey_data_raw - minvoltage) / (maxvoltage - minvoltage)
    S_v = R_v * (maxrange - minrange) + minrange
    meg_gazex_data_scaled = S_h * (screenright - screenleft + 1) + screenleft
    meg_gazey_data_scaled = S_v * (screenbottom - screentop + 1) + screentop

    return meg_gazex_data_scaled, meg_gazey_data_scaled


def blinks_to_nan(exp_info, subject, meg_pupils_data_raw, meg_gazex_data_scaled, meg_gazey_data_scaled):
    print('Removing blinks')

    # Get configuration
    config = subject.params.preproc
    pupil_size_thresh = config.pupil_thresh
    start_interval_samples = config.start_interval_samples
    end_interval_samples = config.end_interval_samples

    # Copy pupils data to detect blinks from
    meg_gazex_data_clean = copy.copy(meg_gazex_data_scaled)
    meg_gazey_data_clean = copy.copy(meg_gazey_data_scaled)
    meg_pupils_data_clean = copy.copy(meg_pupils_data_raw)

    # Define missing values as 1 and non-missing as 0 instead of True False
    if subject.subject_id in exp_info.no_pupil_subjects:
        missing = (meg_gazex_data_clean < pupil_size_thresh).astype(int)
    else:
        missing = (meg_pupils_data_clean < pupil_size_thresh).astype(int)

    # Get missing start/end samples and duration
    missing_start = np.where(np.diff(missing) == 1)[0]
    missing_end = np.where(np.diff(missing) == -1)[0]

    # Take samples according to start or end with missing data (missing_start and end would have different sizes)
    if missing[0] == 1:
        missing_end = missing_end[1:]
    if missing[-1] == 1:
        missing_start = missing_start[:-1]

    # Remove blinks by filling missing values (and suroundings) with nan
    for i in range(len(missing_start)):
        blink_interval = np.arange(missing_start[i] - start_interval_samples, missing_end[i] + end_interval_samples)
        meg_gazex_data_clean[blink_interval] = float('nan')
        meg_gazey_data_clean[blink_interval] = float('nan')
        meg_pupils_data_clean[blink_interval] = float('nan')

    # Check for peaks in pupils signal
    n_peaks_idx, _ = sgn.find_peaks(-meg_pupils_data_clean, prominence=0.15, width=[0, 160])
    p_peaks_idx, _ = sgn.find_peaks(meg_pupils_data_clean, prominence=0.15, width=[0, 160])

    for i in range(len(n_peaks_idx)):
        peak_interval = np.arange(n_peaks_idx[i] - start_interval_samples, n_peaks_idx[i] + end_interval_samples)
        meg_gazex_data_clean[peak_interval] = float('nan')
        meg_gazey_data_clean[peak_interval] = float('nan')
        meg_pupils_data_clean[peak_interval] = float('nan')

    for i in range(len(p_peaks_idx)):
        peak_interval = np.arange(p_peaks_idx[i] - start_interval_samples, p_peaks_idx[i] + end_interval_samples)
        meg_gazex_data_clean[peak_interval] = float('nan')
        meg_gazey_data_clean[peak_interval] = float('nan')
        meg_pupils_data_clean[peak_interval] = float('nan')

    et_channels_meg = [meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean]

    return et_channels_meg


def DAC_samples(et_channels_meg, exp_info, sfreq):
    print('Compensating DAC delay')
    time_delay = exp_info.DAC_delay
    samples_delay = int(round(time_delay / 1000 * sfreq, 0))

    for i in range(len(et_channels_meg)):
        et_channels_meg[i] = np.concatenate((et_channels_meg[i][samples_delay:], np.zeros(samples_delay)))

    # Get separate data from et channels
    meg_gazex_data_raw = et_channels_meg[0]
    meg_gazey_data_raw = et_channels_meg[1]
    meg_pupils_data_raw = et_channels_meg[2]

    return meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw


def fixations_saccades_detection(raw, et_channels_meg, subject, sac_max_vel=1500, fix_max_amp=1.5, screen_resolution=1920, force_run=False):
    """
       Detects fixations, saccades, and pursuit eye movements from MEG data.

       Parameters
       ----------
       raw : instance of Raw
           The raw MEG data.
       et_channels_meg : list of ndarray
           A list containing the cleaned MEG gaze x, y, and pupil data.
       subject : Subject
           An instance containing subject-specific information.
       sac_max_vel : int, optional
           Maximum saccade velocity threshold (default is 1500).
       fix_max_amp : float, optional
           Maximum fixation amplitude threshold (default is 1.5).
       screen_resolution : int, optional
           Screen resolution in pixels (default is 1920).
       force_run : bool, optional
           Force the detection to run even if precomputed data exists (default is False).

       Returns
       -------
       fixations : DataFrame
           Detected fixations with mean positions, pupil size, and adjacent events.
       saccades : DataFrame
           Detected saccades after applying the velocity threshold.
       pursuit : DataFrame
           Detected pursuit movements.
       subject : Subject
           The subject instance with updated counts of detected movements.

       Notes
       -----
       This function uses the Remodnav tool to detect eye movements and then processes
       the detected movements to filter out undesired saccades and fixations based on
       specified thresholds. The function also calculates the average pupil size and
       gaze positions during each fixation.

       Example
       -------
       >>> raw = ...  # Load raw MEG data
       >>> et_channels_meg = [gazex_data, gazey_data, pupil_data]
       >>> subject = setup.raw_subject(exp_info=exp_info, config=config, subject_code=subject_code)
       >>> fixations, saccades, pursuit, subject = fixations_saccades_detection(raw, et_channels_meg, subject)
       """

    out_fname = f'Fix_Sac_detection_{subject.subject_id}.tsv'
    out_folder = paths.preproc_path + subject.subject_id + '/Sac-Fix_detection/'

    meg_gazex_data_clean = et_channels_meg[0]
    meg_gazey_data_clean = et_channels_meg[1]
    meg_pupils_data_clean = et_channels_meg[2]

    if not force_run:
        try:
            # Load pre run saccades and fixation detection
            eye_movements = pd.read_csv(out_folder + out_fname, sep='\t')
            print('\nEye movements loaded')
        except:
            force_run = True

    if force_run:
        # If not pre run data, run
        print('\nRunning eye movements detection')

        # Define data to save to excel file needed to run the saccades detection program Remodnav
        eye_data = {'x': meg_gazex_data_clean, 'y': meg_gazey_data_clean}
        df = pd.DataFrame(eye_data)

        # Remodnav parameters
        fname = f'eye_data_{subject.subject_id}.csv'
        px2deg = math.degrees(math.atan2(.5 * subject.screen_size, subject.screen_distance)) / (.5 * screen_resolution)
        sfreq = raw.info['sfreq']

        # Save csv file
        df.to_csv(fname, sep='\t', header=False, index=False)

        # Run Remodnav not considering pursuit class and min fixations 100 ms
        command = f'remodnav {fname} {out_fname} {px2deg} {sfreq} ' \
                  f'--savgol-length {0.0195} --min-pursuit-duration {0.5} --max-pso-duration {0.05} --min-fixation-duration {0.0} --max-vel {5000}'
        os.system(command)

        # Read results file with detections
        eye_movements = pd.read_csv(out_fname, sep='\t')

        # Move eye data, detections file and image to subject results directory
        os.makedirs(out_folder, exist_ok=True)
        # Move et data file
        os.replace(fname, out_folder + fname)
        # Move results file
        os.replace(out_fname, out_folder + out_fname)
        # Move results image
        out_fname = out_fname.replace('tsv', 'png')
        os.replace(out_fname, out_folder + out_fname)

    # Get saccades, fixations and pursuit
    saccades_all = copy.copy(eye_movements.loc[(eye_movements['label'] == 'SACC') | (eye_movements['label'] == 'ISAC')])
    fixations_all = copy.copy(eye_movements.loc[eye_movements['label'] == 'FIXA'])
    pursuit = copy.copy(eye_movements.loc[eye_movements['label'] == 'PURS'])

    # Drop saccades and fixations based on conditions
    fixations = copy.copy(fixations_all[(fixations_all['amp'] <= fix_max_amp)])
    print(f'Dropping saccades with average vel > {sac_max_vel}, and fixations with amplitude > {fix_max_amp}')

    saccades = copy.copy(saccades_all[saccades_all['peak_vel'] <= sac_max_vel])
    print(f'Kept {len(fixations)} out of {len(fixations_all)} fixations')
    print(f'Kept {len(saccades)} out of {len(saccades_all)} saccades')

    subject.len_all_sac = len(saccades_all)
    subject.len_all_fix = len(fixations_all)
    subject.len_sac_drop = len(saccades)
    subject.len_fix_drop = len(fixations)

    times = raw.times
    mean_x = []
    mean_y = []
    pupil_size = []
    prev_evt = []
    next_evt = []

    # ---------------- Find previous and next event of every fixation ----------------#
    # Remove fixations from and to blink
    print('Finding previous and next events')
    i = 0
    for fix_idx, fixation in fixations.iterrows():

        fix_time = fixation['onset']
        fix_dur = fixation['duration']

        time_thresh = 0.002  # 2 ms
        # Previous and next saccades
        try:
            evt0 = saccades.loc[
                (saccades['onset'] + saccades['duration'] > fix_time - time_thresh) & (saccades['onset'] + saccades['duration'] < fix_time + time_thresh)].index.values[
                -1]
        except:
            try:
                evt0 = pursuit.loc[
                    (pursuit['onset'] + pursuit['duration'] > fix_time - time_thresh) & (pursuit['onset'] + pursuit['duration'] < fix_time + time_thresh)].index.values[
                    -1]
            except:
                evt0 = None
        prev_evt.append(evt0)

        try:
            evt1 = saccades.loc[(saccades['onset'] > fix_time + fix_dur - time_thresh) & (saccades['onset'] < fix_time + fix_dur + time_thresh)].index.values[0]
        except:
            try:
                evt1 = pursuit.loc[(pursuit['onset'] > fix_time + fix_dur - time_thresh) & (pursuit['onset'] < fix_time + fix_dur + time_thresh)].index.values[0]
            except:
                evt1 = None
        next_evt.append(evt1)

        print("\rProgress: {}%".format(int((i + 1) * 100 / len(fixations))), end='')
        i += 1

    # Add columns
    fixations['prev_evt'] = prev_evt
    fixations['next_evt'] = next_evt

    # Drop when None
    fixations.dropna(subset=['prev_evt'], inplace=True)
    print(f'\nKept {len(fixations)} fixations with previous event')

    # ---------------- Average pupil size in each fixation ----------------#
    print('Computing average pupil size, and x and y position')
    i = 0
    for fix_idx, fixation in fixations.iterrows():
        fix_time = fixation['onset']
        fix_dur = fixation['duration']

        # Average pupil size, x and y position
        fix_time_idx = np.where(np.logical_and(fix_time < times, times < fix_time + fix_dur))[0]

        pupil_data_fix = meg_pupils_data_clean[fix_time_idx]
        gazex_data_fix = meg_gazex_data_clean[fix_time_idx]
        gazey_data_fix = meg_gazey_data_clean[fix_time_idx]

        pupil_size.append(np.nanmean(pupil_data_fix))
        mean_x.append(np.nanmean(gazex_data_fix))
        mean_y.append(np.nanmean(gazey_data_fix))

        print("\rProgress: {}%".format(int((i + 1) * 100 / len(fixations))), end='')
        i += 1

    # ---------------- Save to fixations DataFrame ----------------#
    print()
    fixations['mean_x'] = mean_x
    fixations['mean_y'] = mean_y
    fixations['pupil'] = pupil_size
    fixations = fixations.astype({'mean_x': float, 'mean_y': float, 'pupil': float, 'prev_evt': 'Int64', 'next_evt': 'Int64'})

    # ---------------- Compute saccades direction ----------------#
    sac_deg = []
    sac_dir = []
    for row_idx, saccade in saccades.iterrows():

        # Saccades degree
        x_dif = saccade['end_x'] - saccade['start_x']
        y_dif = saccade['end_y'] - saccade['start_y']
        z = complex(x_dif, y_dif)
        deg = np.angle(z, deg=True)
        sac_deg.append(deg)

        if -15 < deg < 15:
            dir = 'r'
        elif 75 < deg < 105:
            dir = 'd'
        elif 165 < deg or deg < -165:
            dir = 'l'
        elif -75 > deg > -105:
            dir = 'u'
        else:
            dir = 'none'
        sac_dir.append(dir)

    # Append to DataFrame
    saccades['deg'] = sac_deg
    saccades['dir'] = sac_dir

    # Save to subject
    subject.fixations = fixations
    subject.saccades = saccades
    subject.pursuit = pursuit

    return fixations, saccades, pursuit, subject


def define_trials_trig(raw, exp_info):
    """
    Raw MEG data has only button presses as annotations. We want to include also trial/eyemap start/end times as events

    Define events (responses) and trials start/end from triggers
    make lists containing response times, trial start times, and trial end times, eyemap start times, eyemap end times, block start /times
    append those lists to raw.annotations giving proper names to each event and list.


    Parameters
    ----------
    raw
    exp_info

    Returns
    -------
    raw

    """
    print('Adding trial onsets from tiggers as annotations...')

    # Extract trigger channel
    trig_ch = raw.copy().pick(exp_info.trig_ch)

    # Get button and triggers values
    trig_values = trig_ch.get_data().ravel()
    trig_dif = np.diff(trig_values)

    # Get time array
    meg_times = trig_ch.times

    # Experiment start
    exp_start_onset = meg_times[np.where(trig_dif == 16)[0]]
    exp_start_desc = ['exp_start']

    # Emaps start
    emap_start_onset = meg_times[np.where(trig_dif == 48)[0]]
    emap_start_desc = [f'emap_bl_1', f'emap_hl_1', f'emap_hs_1', f'emap_vl_1', f'emap_vs_1', f'emap_pur_1',
                       f'emap_bl_2', f'emap_hl_2', f'emap_hs_2', f'emap_vl_2', f'emap_vs_2', f'emap_pur_2',
                       f'emap_bl_3', f'emap_hl_3', f'emap_hs_3', f'emap_vl_3', f'emap_vs_3', f'emap_pur_3']

    # Pre-trial pause start
    pause_start_onset = meg_times[np.where(trig_dif == 64)[0]]
    pause_start_desc = [f'pause_start_{i + 1}' for i in range(len(pause_start_onset))]

    # Eyemap end
    emap_end_onset = pause_start_onset[[0, 11, 22]]
    emap_end_desc = [f'emap_end_{i + 1}' for i in range(len(emap_end_onset))]

    # Trial start
    trial_start_onset = list(meg_times[np.where(trig_dif == 80)[0]])
    trial_start_desc = [f'trial_start_{i + 1}' for i in range(len(trial_start_onset))]

    # Block end
    block_end_onset = meg_times[np.where(trig_dif == 96)[0]]
    block_end_desc = [f'block_end_{i + 1}' for i in range(len(block_end_onset))]

    # Trial end
    trial_end_onset = list(pause_start_onset[1:11]) + list(pause_start_onset[12:22]) + list(pause_start_onset[23:33]) + list(block_end_onset)
    trial_end_onset.sort()
    trial_end_desc = [f'trial_end_{i + 1}' for i in range(len(trial_end_onset))]

    # Onset lists
    onset_lists = [exp_start_onset, emap_start_onset, emap_end_onset, trial_start_onset, trial_end_onset, block_end_onset]
    desc_lists = [exp_start_desc, emap_start_desc, emap_end_desc, trial_start_desc, trial_end_desc, block_end_desc]

    # Get annotations from MEG
    annot_onset = raw.annotations.onset
    annot_desc = raw.annotations.description

    # Iterate concatenating annotations
    for onset, desc in zip(onset_lists, desc_lists):
        annot_onset = np.concatenate((annot_onset, onset))
        annot_desc = np.concatenate((annot_desc, desc))

    # Order annotations chronologically
    raw_annot = np.array([annot_onset, annot_desc])
    raw_annot = raw_annot[:, raw_annot[0].astype(float).argsort()]

    # Update annotations in MEG
    raw.annotations.onset = raw_annot[0].astype(float)
    raw.annotations.description = raw_annot[1]

    # Set durations and ch_names length to match the annotations length
    raw.annotations.duration = np.zeros(len(raw.annotations.description))

    return raw


def saccades_classification(subject, saccades, raw):
    print('\nClassifying saccades')

    # Eyemap trials start and end indexes
    hl_start_idx = np.where(raw.annotations.description == 'hl_start')[0]
    vl_start_idx = np.where(raw.annotations.description == 'vl_start')[0]
    hs_start_idx = np.where(raw.annotations.description == 'hs_start')[0]
    vs_start_idx = np.where(raw.annotations.description == 'vs_start')[0]
    bl_start_idx = np.where(raw.annotations.description == 'bl_start')[0]
    bl_end_idx = np.where(raw.annotations.description == 'bl_end')[0]

    # Eyemap trials start and end times
    hl_start_times = raw.annotations.onset[hl_start_idx]
    vl_start_times = raw.annotations.onset[vl_start_idx]
    hs_start_times = raw.annotations.onset[hs_start_idx]
    vs_start_times = raw.annotations.onset[vs_start_idx]
    bl_start_times = raw.annotations.onset[bl_start_idx]
    bl_end_times = raw.annotations.onset[bl_end_idx]

    response_trials_meg = subject.trial
    cross1_times_meg = subject.cross1
    ms_times_meg = subject.ms
    cross2_times_meg = subject.cross2
    vs_times_meg = subject.vs
    vsend_times_meg = subject.vsend

    # Get mss and target pres/abs from bh data
    mss = subject.bh_data['Nstim'].astype(int)
    pres_abs = subject.bh_data['Tpres'].astype(int)

    # Determine corr ans from mapping and meg response
    corr_ans = np.zeros(len(subject.bh_data)).astype(int)
    corr_answers = subject.bh_data['corrAns']
    actual_answers = subject.description
    for i, trial in enumerate(response_trials_meg):
        trial_idx = trial - 1
        if int(subject.buttons_pc_map[actual_answers[i]]) != corr_answers[trial_idx]:  # Check if mapping of button corresponds to correct answer
            corr_ans[trial_idx] = 0
        else:
            corr_ans[trial_idx] = 1

    # Add correct ans data to subject
    subject.corr_ans = corr_ans

    # Get MSS, present/absent for every trial
    sac_trial = []
    sac_screen = []
    trial_mss = []
    tgt_pres_abs = []
    trial_correct = []
    n_sacs = []
    sac_delay = []
    sac_deg = []
    sac_dir = []
    sac_id = []
    description = []
    onset = []

    # Define dict to store screen-trial fixation number
    sac_numbers = {}

    i = 0
    for row_idx, saccade in saccades.iterrows():
        sac_time = saccade['onset']

        # Saccades degree
        x_dif = saccade['end_x'] - saccade['start_x']
        y_dif = saccade['end_y'] - saccade['start_y']
        z = complex(x_dif, y_dif)
        deg = np.angle(z, deg=True)
        sac_deg.append(deg)

        if -15 < deg < 15:
            dir = 'r'
        elif 75 < deg < 105:
            dir = 'd'
        elif 165 < deg or deg < -165:
            dir = 'l'
        elif -75 > deg > -105:
            dir = 'u'
        else:
            dir = 'none'
        sac_dir.append(dir)

        # find saccades block
        emap_sac = False
        for (block_num, emap_start_time), emap_end_time in zip(enumerate(hl_start_times), bl_end_times):
            if emap_start_time < sac_time < emap_end_time:
                emap_sac = True
                break

        # Save sac number within block
        if block_num not in sac_numbers.keys():
            sac_numbers[block_num] = {'emap_hl': 0, 'emap_vl': 0, 'emap_hs': 0, 'emap_vs': 0, 'emap_bl': 0}

        if emap_sac:
            # Find saccades emap trial
            sac_trial.append(None)
            trial_mss.append(None)
            tgt_pres_abs.append(None)
            trial_correct.append(None)

            # hl
            if hl_start_times[block_num] < sac_time < vl_start_times[block_num]:
                # Saccade data
                screen = 'emap_hl'
                sac_screen.append(screen)
                sac_delay.append(sac_time - hl_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                id = f'{dir}_sac_{screen}{block_num}_{n_sacs[-1]}'
                sac_id.append(id)
                description.append(id)
                onset.append([sac_time])
            # vl
            elif vl_start_times[block_num] < sac_time < hs_start_times[block_num]:
                # Saccade data
                screen = 'emap_vl'
                sac_screen.append(screen)
                sac_delay.append(sac_time - vl_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                id = f'{dir}_sac_{screen}{block_num}_{n_sacs[-1]}'
                sac_id.append(id)
                description.append(id)
                onset.append([sac_time])
            # hs
            elif hs_start_times[block_num] < sac_time < vs_start_times[block_num]:
                # Saccade data
                screen = 'emap_hs'
                sac_screen.append(screen)
                sac_delay.append(sac_time - hs_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                id = f'{dir}_sac_{screen}{block_num}_{n_sacs[-1]}'
                sac_id.append(id)
                description.append(id)
                onset.append([sac_time])
            # vs
            elif vs_start_times[block_num] < sac_time < bl_start_times[block_num]:
                # Saccade data
                screen = 'emap_vs'
                sac_screen.append(screen)
                sac_delay.append(sac_time - vs_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                id = f'{dir}_sac_{screen}{block_num}_{n_sacs[-1]}'
                sac_id.append(id)
                description.append(id)
                onset.append([sac_time])
            # bl
            elif bl_start_times[block_num] < sac_time < bl_end_times[block_num]:
                # Saccade data
                screen = 'emap_bl'
                sac_screen.append(screen)
                sac_delay.append(sac_time - bl_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                id = f'{dir}_sac_{screen}{block_num}_{n_sacs[-1]}'
                sac_id.append(id)
                description.append(id)
                onset.append([sac_time])

            else:
                sac_screen.append(None)
                sac_delay.append(None)
                n_sacs.append(None)
                sac_id.append(None)

        # No emap saccade
        else:
            trial_found = False
            for (trial_idx, trial_start_time), trial_end_time in zip(enumerate(cross1_times_meg), vsend_times_meg):
                if trial_start_time < sac_time < trial_end_time:
                    trial_found = True
                    break

            if trial_found:
                # Define trial to store screen sacation number. if sac_time > all trials end time, sac gets stored as last trial under screen None.
                trial = trial_idx + 1

                sac_trial.append(trial)
                trial_mss.append(mss[trial_idx])
                tgt_pres_abs.append(pres_abs[trial_idx])
                trial_correct.append(corr_ans[trial_idx])

                if f'trial_{trial}' not in sac_numbers[block_num].keys():
                    sac_numbers[block_num][f'trial_{trial}'] = {'cross1': 0, 'ms': 0, 'cross2': 0, 'vs': 0}

                # First sacation cross
                if cross1_times_meg[trial_idx] < sac_time < ms_times_meg[trial_idx]:
                    # Saccade data
                    screen = 'cross1'
                    sac_screen.append(screen)
                    sac_delay.append(sac_time - cross1_times_meg[trial_idx])
                    sac_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_sacs.append(sac_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    id = f'{dir}_sac_{screen}_t{trial}_{n_sacs[-1]}'
                    sac_id.append(id)
                    description.append(id)
                    onset.append([sac_time])

                # MS
                elif ms_times_meg[trial_idx] < sac_time < cross2_times_meg[trial_idx]:
                    # Saccade data
                    screen = 'ms'
                    sac_screen.append(screen)
                    sac_delay.append(sac_time - ms_times_meg[trial_idx])
                    sac_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_sacs.append(sac_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    id = f'{dir}_sac_{screen}_t{trial}_{n_sacs[-1]}'
                    sac_id.append(id)
                    description.append(id)
                    onset.append([sac_time])

                # Second sacations corss
                elif cross2_times_meg[trial_idx] < sac_time < vs_times_meg[trial_idx]:
                    # Saccade data
                    screen = 'cross2'
                    sac_screen.append(screen)
                    sac_delay.append(sac_time - cross2_times_meg[trial_idx])
                    sac_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_sacs.append(sac_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    id = f'{dir}_sac_{screen}_t{trial}_{n_sacs[-1]}'
                    sac_id.append(id)
                    description.append(id)
                    onset.append([sac_time])

                # VS
                elif vs_times_meg[trial_idx] < sac_time < vsend_times_meg[trial_idx]:
                    # Saccade data
                    screen = 'vs'
                    sac_screen.append(screen)
                    sac_delay.append(sac_time - vs_times_meg[trial_idx])
                    sac_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_sacs.append(sac_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    id = f'{dir}_sac_{screen}_t{trial}_{n_sacs[-1]}'
                    sac_id.append(id)
                    description.append(id)
                    onset.append([sac_time])

                # No screen identified
                else:
                    sac_screen.append(None)
                    sac_delay.append(None)
                    n_sacs.append(None)
                    sac_id.append(None)

            # No trial identified
            else:
                sac_trial.append(None)
                trial_mss.append(None)
                tgt_pres_abs.append(None)
                trial_correct.append(None)
                sac_screen.append(None)
                sac_delay.append(None)
                n_sacs.append(None)
                sac_id.append(None)

        print("\rProgress: {}%".format(int((i + 1) * 100 / len(saccades))), end='')
        i += 1

    print()
    saccades['trial'] = sac_trial
    saccades['mss'] = trial_mss
    saccades['screen'] = sac_screen
    saccades['target_pres'] = tgt_pres_abs
    saccades['correct'] = trial_correct
    saccades['n_sac'] = n_sacs
    saccades['delay'] = sac_delay
    saccades['deg'] = sac_deg
    saccades['dir'] = sac_dir
    saccades['id'] = sac_id

    # Rearange columns
    col1 = saccades.pop('deg')
    saccades.insert(8, col1.name, col1)
    col2 = saccades.pop('dir')
    saccades.insert(9, col2.name, col2)

    # Define column type
    saccades = saccades.astype({'trial': 'Int64', 'mss': 'Int64', 'target_pres': 'Int64', 'delay': float,
                                'correct': 'Int64', 'n_sac': 'Int64', 'deg': float, 'dir': str, 'id': str})

    # Save to subject
    subject.saccades = saccades

    # Add vs fixations data to raw annotations
    raw.annotations.description = np.concatenate((raw.annotations.description, np.array(description)))
    raw.annotations.onset = np.concatenate((raw.annotations.onset, np.array(functions_general.flatten_list(onset))))

    # Order raw.annotations chronologically
    raw_annot = np.array([raw.annotations.onset, raw.annotations.description])
    raw_annot = raw_annot[:, raw_annot[0].astype(float).argsort()]
    raw.annotations.onset = raw_annot[0].astype(float)
    raw.annotations.description = raw_annot[1]

    # Set durations and ch_names length to match the annotations length
    raw.annotations.duration = np.zeros(len(raw.annotations.description))

    return saccades, raw, subject


def fixation_classification(subject, fixations, raw):
    # Eyemap trials start and end indexes
    hl_start_idx = np.where(raw.annotations.description == 'hl_start')[0]
    vl_start_idx = np.where(raw.annotations.description == 'vl_start')[0]
    hs_start_idx = np.where(raw.annotations.description == 'hs_start')[0]
    vs_start_idx = np.where(raw.annotations.description == 'vs_start')[0]
    bl_start_idx = np.where(raw.annotations.description == 'bl_start')[0]
    bl_end_idx = np.where(raw.annotations.description == 'bl_end')[0]

    # Eyemap trials start and end times
    hl_start_times = raw.annotations.onset[hl_start_idx]
    vl_start_times = raw.annotations.onset[vl_start_idx]
    hs_start_times = raw.annotations.onset[hs_start_idx]
    vs_start_times = raw.annotations.onset[vs_start_idx]
    bl_start_times = raw.annotations.onset[bl_start_idx]
    bl_end_times = raw.annotations.onset[bl_end_idx]

    cross1_times_meg = subject.cross1
    ms_times_meg = subject.ms
    cross2_times_meg = subject.cross2
    vs_times_meg = subject.vs
    vsend_times_meg = subject.vsend

    # Get mss and target pres/abs from bh data
    mss = subject.bh_data['Nstim'].astype(int)
    pres_abs = subject.bh_data['Tpres'].astype(int)

    # Get corr ans from meg event and mapping
    corr_ans = subject.corr_ans

    # Get MSS, present/absent for every trial
    fix_trial = []
    fix_screen = []
    trial_mss = []
    tgt_pres_abs = []
    trial_correct = []
    n_fixs = []
    fix_delay = []
    fix_id = []

    description = []
    onset = []

    # Iterate over fixations to classify them
    print('Classifying fixations')

    # Define dict to store screen-trial fixation number
    fix_numbers = {}

    i = 0
    for fix_idx, fixation in fixations.iterrows():

        fix_time = fixation['onset']

        emap_fix = False

        # find fixation's block
        for (block_num, emap_start_time), emap_end_time in zip(enumerate(hl_start_times), bl_end_times):
            if emap_start_time < fix_time < emap_end_time:
                emap_fix = True
                break

        # Save fix number within block
        if block_num not in fix_numbers.keys():
            fix_numbers[block_num] = {'emap_hl': 0, 'emap_vl': 0, 'emap_hs': 0, 'emap_vs': 0, 'emap_bl': 0}

        if emap_fix:
            # Find fixations emap trial
            fix_trial.append(None)
            trial_mss.append(None)
            tgt_pres_abs.append(None)
            trial_correct.append(None)

            # hl
            if hl_start_times[block_num] < fix_time < vl_start_times[block_num]:
                # fixation data
                screen = 'emap_hl'
                fix_screen.append(screen)
                fix_delay.append(fix_time - hl_start_times[block_num])
                fix_numbers[block_num][screen] += 1
                n_fixs.append(fix_numbers[block_num][screen])

                # Save to raw
                id = f'fix_{screen}{block_num}_{n_fixs[-1]}'
                fix_id.append(id)
                description.append(id)
                onset.append([fix_time])
            # vl
            elif vl_start_times[block_num] < fix_time < hs_start_times[block_num]:
                # fixation data
                screen = 'emap_vl'
                fix_screen.append(screen)
                fix_delay.append(fix_time - vl_start_times[block_num])
                fix_numbers[block_num][screen] += 1
                n_fixs.append(fix_numbers[block_num][screen])

                # Save to raw
                id = f'fix_{screen}{block_num}_{n_fixs[-1]}'
                fix_id.append(id)
                description.append(id)
                onset.append([fix_time])
            # hs
            elif hs_start_times[block_num] < fix_time < vs_start_times[block_num]:
                # fixation data
                screen = 'emap_hs'
                fix_screen.append(screen)
                fix_delay.append(fix_time - hs_start_times[block_num])
                fix_numbers[block_num][screen] += 1
                n_fixs.append(fix_numbers[block_num][screen])

                # Save to raw
                id = f'fix_{screen}{block_num}_{n_fixs[-1]}'
                fix_id.append(id)
                description.append(id)
                onset.append([fix_time])
            # vs
            elif vs_start_times[block_num] < fix_time < bl_start_times[block_num]:
                # fixation data
                screen = 'emap_vs'
                fix_screen.append(screen)
                fix_delay.append(fix_time - vs_start_times[block_num])
                fix_numbers[block_num][screen] += 1
                n_fixs.append(fix_numbers[block_num][screen])

                # Save to raw
                id = f'fix_{screen}{block_num}_{n_fixs[-1]}'
                fix_id.append(id)
                description.append(id)
                onset.append([fix_time])
            # bl
            elif bl_start_times[block_num] < fix_time < bl_end_times[block_num]:
                # fixation data
                screen = 'emap_bl'
                fix_screen.append(screen)
                fix_delay.append(fix_time - bl_start_times[block_num])
                fix_numbers[block_num][screen] += 1
                n_fixs.append(fix_numbers[block_num][screen])

                # Save to raw
                id = f'fix_{screen}{block_num}_{n_fixs[-1]}'
                fix_id.append(id)
                description.append(id)
                onset.append([fix_time])

            else:
                # No emap trial identified
                fix_screen.append(None)
                fix_delay.append(None)
                n_fixs.append(None)
                fix_id.append(None)

        # No emap fixation
        else:
            trial_found = False
            for (trial_idx, trial_start_time), trial_end_time in zip(enumerate(cross1_times_meg), vsend_times_meg):
                if trial_start_time < fix_time < trial_end_time:
                    trial_found = True
                    break

            if trial_found:
                # Define trial to store screen fixation number. if fix_time > all trials end time, fix gets stored as last trial under screen None.
                trial = trial_idx + 1

                fix_trial.append(trial)
                trial_mss.append(mss[trial_idx])
                tgt_pres_abs.append(pres_abs[trial_idx])
                trial_correct.append(corr_ans[trial_idx])

                if f'trial_{trial}' not in fix_numbers[block_num].keys():
                    fix_numbers[block_num][f'trial_{trial}'] = {'cross1': 0, 'ms': 0, 'cross2': 0, 'vs': 0}

                # First fixation cross
                if cross1_times_meg[trial_idx] < fix_time < ms_times_meg[trial_idx]:
                    # fixation data
                    screen = 'cross1'
                    fix_screen.append(screen)
                    fix_delay.append(fix_time - cross1_times_meg[trial_idx])
                    fix_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_fixs.append(fix_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    id = f'fix_{screen}_t{trial}_{n_fixs[-1]}'
                    fix_id.append(id)
                    description.append(id)
                    onset.append([fix_time])

                # MS
                elif ms_times_meg[trial_idx] < fix_time < cross2_times_meg[trial_idx]:
                    # fixation data
                    screen = 'ms'
                    fix_screen.append(screen)
                    fix_delay.append(fix_time - ms_times_meg[trial_idx])
                    fix_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_fixs.append(fix_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    id = f'fix_{screen}_t{trial}_{n_fixs[-1]}'
                    fix_id.append(id)
                    description.append(id)
                    onset.append([fix_time])

                # Second fixations corss
                elif cross2_times_meg[trial_idx] < fix_time < vs_times_meg[trial_idx]:
                    # fixation data
                    screen = 'cross2'
                    fix_screen.append(screen)
                    fix_delay.append(fix_time - cross2_times_meg[trial_idx])
                    fix_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_fixs.append(fix_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    id = f'fix_{screen}_t{trial}_{n_fixs[-1]}'
                    fix_id.append(id)
                    description.append(id)
                    onset.append([fix_time])

                # VS
                elif vs_times_meg[trial_idx] < fix_time < vsend_times_meg[trial_idx]:
                    # fixation data
                    screen = 'vs'
                    fix_screen.append(screen)
                    fix_delay.append(fix_time - vs_times_meg[trial_idx])
                    fix_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_fixs.append(fix_numbers[block_num][f'trial_{trial}'][screen])
                    # Will specify id in target classification
                    fix_id.append(None)

                # No screen identified
                else:
                    fix_screen.append(None)
                    fix_delay.append(None)
                    n_fixs.append(None)
                    fix_id.append(None)

            # No trial identified
            else:
                fix_trial.append(None)
                trial_mss.append(None)
                tgt_pres_abs.append(None)
                trial_correct.append(None)
                fix_screen.append(None)
                fix_delay.append(None)
                n_fixs.append(None)
                fix_id.append(None)

        print("\rProgress: {}%".format(int((i + 1) * 100 / len(fixations))), end='')
        i += 1

    print()
    fixations['trial'] = fix_trial
    fixations['mss'] = trial_mss
    fixations['screen'] = fix_screen
    fixations['target_pres'] = tgt_pres_abs
    fixations['correct'] = trial_correct
    fixations['n_fix'] = n_fixs
    fixations['delay'] = fix_delay
    fixations['id'] = fix_id

    fixations = fixations.astype({'trial': 'Int64', 'mss': 'Int64', 'target_pres': 'Int64', 'delay': float, 'correct': 'Int64',
                                  'n_fix': 'Int64', 'id': str})

    # Add vs fixations data to raw annotations
    raw.annotations.description = np.concatenate((raw.annotations.description, np.array(description)))
    raw.annotations.onset = np.concatenate((raw.annotations.onset, np.array(functions_general.flatten_list(onset))))

    return fixations, raw


def target_vs_distractor(fixations, subject, raw, distance_threshold=70, screen_res_x=1920, screen_res_y=1080,
                         img_res_x=1280, img_res_y=1024):
    print('Identifying fixated items in VS screen')

    # Load items data
    items_pos_path = paths.item_pos_path
    items_pos = pd.read_csv(items_pos_path)

    # Rescale images from original resolution to screen resolution to match fixations scale
    items_pos['pos_x_corr'] = items_pos['pos_x'] + (screen_res_x - img_res_x) / 2
    items_pos['center_x_corr'] = items_pos['center_x'] + (screen_res_x - img_res_x) / 2
    items_pos['pos_y_corr'] = items_pos['pos_y'] + (screen_res_y - img_res_y) / 2
    items_pos['center_y_corr'] = items_pos['center_y'] + (screen_res_y - img_res_y) / 2

    # iterate over fixations checking for trial number, then check image used, then check in item_pos the position of items and mesure distance
    fixations_vs = copy.copy(fixations.loc[fixations['screen'] == 'vs'])

    # Define save data variables
    items = []
    fix_item_distance = []
    fix_target = []
    trials_image = []
    fix_id = []

    description = []
    onset = []

    # Iterate over fixations
    for fix_idx, fix in fixations_vs.iterrows():

        # Get fixation trial time, and n_fix
        trial = fix['trial']
        trial_idx = trial - 1

        mss = fix['mss']

        n_fix = fix['n_fix']
        fix_time = fix['onset']

        # Get fixations x and y
        fix_x = fix['mean_x']
        fix_y = fix['mean_y']

        # Get trial image
        trial_image = subject.trial_imgs[trial_idx]
        trials_image.append(trial_image)

        # Find item position information for such image
        trial_items = items_pos.loc[items_pos['folder'] == trial_image]

        # Define trial save variables
        distances = []
        target = []

        # Iterate over trial items
        for item_idx, item in trial_items.iterrows():
            # Item position
            item_x = item['center_x_corr']
            item_y = item['center_y_corr']

            # Fixations to item distance
            x_dist = abs(fix_x - item_x)
            y_dist = abs(fix_y - item_y)
            distance = np.sqrt(x_dist ** 2 + y_dist ** 2)

            distances.append(distance)
            target.append(item['istarget'])

        # Closest item to fixation
        min_distance = np.min(np.array(distances))
        min_distance_idx = np.argmin(np.array(distances))

        if min_distance < distance_threshold:
            item = min_distance_idx
            item_distance = min_distance
            istarget = target[min_distance_idx]
        else:
            item = float('nan')
            item_distance = float('nan')
            istarget = float('nan')

        # Save trial data
        items.append(item)
        fix_item_distance.append(item_distance)
        fix_target.append(istarget)

        if ~np.isnan(istarget) and istarget:
            prefix = 'tgt'
        elif ~np.isnan(item):
            prefix = 'it'
        else:
            prefix = 'none'

        id = f'{prefix}_fix_vs_t{trial}_{n_fix}'
        fix_id.append(id)
        description.append(id)
        onset.append([fix_time])

    # Save to fixations_vs df
    fixations.loc[fixations['screen'] == 'vs', 'item'] = items
    fixations.loc[fixations['screen'] == 'vs', 'fix_target'] = fix_target
    fixations.loc[fixations['screen'] == 'vs', 'distance'] = fix_item_distance
    fixations.loc[fixations['screen'] == 'vs', 'trial_image'] = trials_image
    fixations.loc[fixations['screen'] == 'vs', 'id'] = fix_id

    # Save to subject
    subject.fixations = fixations

    # Save to raw annotations
    raw.annotations.description = np.concatenate((raw.annotations.description, np.array(description)))
    raw.annotations.onset = np.concatenate((raw.annotations.onset, np.array(functions_general.flatten_list(onset))))

    raw_annot = np.array([raw.annotations.onset, raw.annotations.description])
    raw_annot = raw_annot[:, raw_annot[0].astype(float).argsort()]
    raw.annotations.onset = raw_annot[0].astype(float)
    raw.annotations.description = raw_annot[1]

    raw.annotations.duration = np.zeros(len(raw.annotations.description))

    return raw, subject, items_pos


def add_et_channels(raw, et_channels_meg, et_channel_names):
    # ---------------- Add scaled data to meg data ----------------#
    print('\nSaving scaled et data to meg raw data structure')
    # copy raw structure
    raw_et = raw.copy()
    # make new raw structure from et channels only
    raw_et = mne.io.RawArray(et_channels_meg, raw_et.pick(et_channel_names).info)
    # change channel names
    for ch_name, new_name in zip(raw_et.ch_names, ['ET_gaze_x', 'ET_gaze_y', 'ET_pupils']):
        raw_et.rename_channels({ch_name: new_name})

    # Pick data from MEG channels and other channels of interest
    channel_idx = mne.pick_types(raw.info, meg=True)
    channel_idx = np.append(channel_idx, mne.pick_channels(raw.info['ch_names'], ['UPPT001', 'UADC001-4123', 'UADC002-4123', 'UADC013-4123']))
    raw.pick(channel_idx)
    # Rename channels removing '-4123'
    raw.rename_channels(functions_general.ch_name_map)

    # save to original raw structure (requires to load data)
    print('Loading MEG data')
    raw.load_data()
    print('Adding new ET channels')
    raw.add_channels([raw_et], force_update_info=True)

    return raw


def filter_line_noise(subject, raw, freqs=(50, 100, 110, 150, 200, 250, 300), display_fig=False, save_fig=True,
                      fig_path=None, fig_name='RAW_PSD'):
    # Pick filter channels
    meg_picks = mne.pick_types(raw.info, meg=True)

    # Filter
    filtered_data = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)

    # Plot
    plot_preproc.line_noise_psd(subject=subject, raw=raw, filtered=filtered_data, display_fig=display_fig,
                                save_fig=save_fig, fig_path=fig_path, fig_name=fig_name)

    return filtered_data


def set_digitlization(subject, meg_data):
    # Load digitalization file
    dig_path = paths.opt_path
    dig_path_subject = dig_path + subject.subject_id
    dig_filepath = dig_path_subject + '/Model_Mesh_5m_headers.pos'
    pos = pd.read_table(dig_filepath, index_col=0)

    # Get fiducials from dig
    nasion = pos.loc[pos.index == 'nasion ']
    lpa = pos.loc[pos.index == 'left ']
    rpa = pos.loc[pos.index == 'right ']

    # Get head points
    pos.drop(['nasion ', 'left ', 'right '], inplace=True)
    pos_array = pos.to_numpy()

    # Make montage
    dig_montage = mne.channels.make_dig_montage(nasion=nasion.values.ravel(), lpa=lpa.values.ravel(),
                                                rpa=rpa.values.ravel(), hsp=pos_array, coord_frame='unknown')

    # Make info object
    meg_data.info.set_montage(montage=dig_montage)

    return meg_data


## OLD out of use


def overwrite_et_channels(raw, et_channels_meg, et_channel_names):
    # ---------------- Add scaled data to meg data ----------------#
    print('\nSaving scaled et data to meg raw data structure')

    # Get et channels idx
    et_ch_idx = mne.pick_channels(raw.info['ch_names'], ['UADC001-4123', 'UADC002-4123', 'UADC013-4123'])

    # Overwrite data in et channels idx for scaled data
    raw.load_data()
    raw._data[et_ch_idx, :] = et_channels_meg

    # Change et channel names
    print('Renaming and droping extra channels')
    for ch_name, new_name in zip(et_channel_names, ['ET_gaze_x', 'ET_gaze_y', 'ET_pupils']):
        raw.rename_channels({ch_name: new_name})

    # Pick data from MEG channels and other channels of interest
    channel_idx = mne.pick_types(raw.info, meg=True)
    channel_idx = np.append(channel_idx, mne.pick_channels(raw.info['ch_names'], ['UPPT001', 'ET_gaze_x', 'ET_gaze_y', 'ET_pupils']))
    raw.pick(channel_idx)

    # Rename channels removing '-4123'
    raw.rename_channels(functions_general.ch_name_map)

    return raw


def fake_blink_interpolate(meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean, sfreq, config):
    print('Interpolating fake blinks')

    # Get interpolation configuration
    blink_min_dur = config.blink_min_dur
    start_interval_samples = config.start_interval_samples
    end_interval_samples = config.end_interval_samples

    # Missing signal
    missing = np.isnan(meg_pupils_data_clean).astype(int)

    # Get missing start/end samples and duration
    missing_start = np.where(np.diff(missing) == 1)[0]
    missing_end = np.where(np.diff(missing) == -1)[0]
    missing_dur = missing_end - missing_start

    # Consider we enlarged the intervals when classifying for real and fake blinks
    blink_min_samples = blink_min_dur / 1000 * sfreq + start_interval_samples + end_interval_samples

    # Get fake blinks based on duration condition (actual blinks were already filled with nan
    fake_blinks = np.where(missing_dur <= blink_min_samples)[0]

    # Interpolate fake blinks
    for fake_blink_idx in fake_blinks:
        blink_interval = np.arange(missing_start[fake_blink_idx] - start_interval_samples,
                                   missing_end[fake_blink_idx] + end_interval_samples)

        interpolation_x = np.linspace(meg_gazex_data_clean[blink_interval[0]], meg_gazex_data_clean[blink_interval[-1]],
                                      len(blink_interval))
        interpolation_y = np.linspace(meg_gazey_data_clean[blink_interval[0]], meg_gazey_data_clean[blink_interval[-1]],
                                      len(blink_interval))
        interpolation_pupil = np.linspace(meg_pupils_data_clean[blink_interval[0]],
                                          meg_pupils_data_clean[blink_interval[-1]], len(blink_interval))
        meg_gazex_data_clean[blink_interval] = interpolation_x
        meg_gazey_data_clean[blink_interval] = interpolation_y
        meg_pupils_data_clean[blink_interval] = interpolation_pupil

    et_channels_meg = [meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean]

    return et_channels_meg


def fake_blink_interpolate2(meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean, sfreq, config):
    print('Interpolating fake blinks')

    # Get interpolation configuration
    blink_min_dur = config.blink_min_dur
    start_interval_samples = config.start_interval_samples
    end_interval_samples = config.end_interval_samples

    # Missing signal
    missing = np.isnan(meg_pupils_data_clean).astype(int)

    # Get missing start/end samples and duration
    missing_start = np.where(np.diff(missing) == 1)[0]
    missing_end = np.where(np.diff(missing) == -1)[0]
    missing_dur = missing_end - missing_start

    missing_dist_x = []
    missing_dist_y = []
    for i in range(len(missing_start)):
        missing_dist_x.append(abs(meg_gazex_data_clean[missing_start[i] - 1] - meg_gazex_data_clean[missing_end[i] + 1]))
        missing_dist_y.append(abs(meg_gazey_data_clean[missing_start[i] - 1] - meg_gazey_data_clean[missing_end[i] + 1]))

    # Consider we enlarged the intervals when classifying for real and fake blinks
    blink_min_samples = blink_min_dur / 1000 * sfreq + start_interval_samples + end_interval_samples
    blink_min_dist = 10

    # Get fake blinks based on duration condition (actual blinks were already filled with nan
    fake_blinks = np.where(np.logical_and((missing_dur <= blink_min_samples), (np.array(missing_dist_x) < blink_min_dist),
                                          (np.array(missing_dist_y) < blink_min_dist)))[0]

    # Interpolate fake blinks
    for fake_blink_idx in fake_blinks:
        blink_interval = np.arange(missing_start[fake_blink_idx] - start_interval_samples,
                                   missing_end[fake_blink_idx] + end_interval_samples)

        interpolation_x = np.linspace(meg_gazex_data_clean[blink_interval[0]], meg_gazex_data_clean[blink_interval[-1]],
                                      len(blink_interval))
        interpolation_y = np.linspace(meg_gazey_data_clean[blink_interval[0]], meg_gazey_data_clean[blink_interval[-1]],
                                      len(blink_interval))
        interpolation_pupil = np.linspace(meg_pupils_data_clean[blink_interval[0]],
                                          meg_pupils_data_clean[blink_interval[-1]], len(blink_interval))
        meg_gazex_data_clean[blink_interval] = interpolation_x
        meg_gazey_data_clean[blink_interval] = interpolation_y
        meg_pupils_data_clean[blink_interval] = interpolation_pupil

    et_channels_meg = [meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean]

    return et_channels_meg


def define_events_trials_BH(raw, subject):
    print('Detecting events and defining trials')

    # Get events in meg data (only red blue and green)
    evt_buttons = raw.annotations.description
    evt_times = raw.annotations.onset[(evt_buttons == 'red') | (evt_buttons == 'blue') | (evt_buttons == 'green')]
    evt_buttons = evt_buttons[(evt_buttons == 'red') | (evt_buttons == 'blue') | (evt_buttons == 'green')]

    # Check for first trial when first response is not green
    first_trial = functions_general.first_trial(evt_buttons)

    # Drop events before 1st trial
    evt_buttons = evt_buttons[first_trial:]
    evt_times = evt_times[first_trial:]

    # Split events into blocks by green press at begening of block
    blocks_start_end = np.where(evt_buttons == 'green')[0]

    # Add first trial idx (0) on first sample (0) and Add last trial idx in the end
    blocks_start_end = np.concatenate((np.array([-1]), blocks_start_end, np.array([len(evt_buttons)])))

    # Delete succesive presses of green
    blocks_start_end = list(np.delete(blocks_start_end, np.where(np.diff(blocks_start_end) < 2)).astype(int))

    # Delete blocks shorter than 20 trials
    blocks_start_end = list(np.delete(blocks_start_end, np.where(np.diff(blocks_start_end) < 20)).astype(int))

    # Define starting and ending trial of each block
    blocks_bounds = [(blocks_start_end[i] + 1, blocks_start_end[i + 1]) for i in range(len(blocks_start_end) - 1)]

    # Load behavioural data
    bh_data = subject.load_raw_bh_data()

    # Get only trial data rows
    bh_data = bh_data.loc[~pd.isna(bh_data['target.started'])].reset_index(drop=True)

    # Get MS start time
    ms_start = bh_data['target.started'].values.ravel().astype(float)

    # Get fix 1 start time
    cross1_start_key_idx = ['fixation_target.' in key and 'started' in key for key in bh_data.keys()]
    cross1_start_key = bh_data.keys()[cross1_start_key_idx][0]

    # If there's a missing cross1 screen, the time for that screen would be None, and the column type would be string.
    # If there's not, the type would be float. Change type to string and replace None by the mss screen start time
    if type(bh_data[cross1_start_key][0]) == str:
        cross1_start = np.array([value.replace('None', f'{ms_start[i]}') for i, value in
                                 enumerate(bh_data[cross1_start_key].values.ravel())]).astype(float)
    else:
        cross1_start = bh_data[cross1_start_key].values

    # Get fix 2 start time
    cross2_start_key_idx = ['fixation_target_2' in key and 'started' in key for key in bh_data.keys()]
    cross2_start_key = bh_data.keys()[cross2_start_key_idx]
    cross2_start = bh_data[cross2_start_key].values.ravel().astype(float)

    # Get Visual search start time
    search_start_key_idx = ['search' in key and 'started' in key for key in bh_data.keys()]
    search_start_key = bh_data.keys()[search_start_key_idx]
    search_start = bh_data[search_start_key].values.ravel().astype(float)

    # Get responses
    responses = bh_data['key_resp.keys'].values

    # Get response time
    rt = copy.copy(bh_data['key_resp.rt'].values)
    # completed_responses_idx = np.where(pd.isna(bh_data['key_resp.rt']) & (responses != 'None'))[0]
    # rt.loc[completed_responses_idx] = 10
    # rt = rt.values

    # Define variables to store data
    no_answer = []
    cross1_times_meg = []
    ms_times_meg = []
    cross2_times_meg = []
    vs_times_meg = []
    buttons_meg = []
    response_times_meg = []
    response_trials_meg = []
    time_differences = []
    description = []
    onset = []

    # Define trials block by block
    for block_num, block_bounds in enumerate(blocks_bounds):
        print(f'\nBlock: {block_num + 1}')
        block_start = block_bounds[0]
        block_end = block_bounds[1]
        block_trials = 30
        block_idxs = np.arange(block_num * block_trials, (block_num + 1) * block_trials)

        # Get events in block from MEG data
        meg_evt_block_times = copy.copy(evt_times[block_start:block_end])
        meg_evt_block_buttons = copy.copy(evt_buttons[block_start:block_end])

        # Get durations in block from BH data
        cross1_block_dur = (ms_start - cross1_start)[block_idxs]
        ms_block_dur = (cross2_start - ms_start)[block_idxs]
        cross2_block_dur = (search_start - cross2_start)[block_idxs]
        rt_block_times = rt[block_idxs]

        # Align MEG and BH data in time
        block_data_aligned = False
        attempts = 0
        align_sample = 0
        while not block_data_aligned and attempts < 5:
            # Save block variables resetting them every attempt
            no_answer_block = []
            cross1_times_meg_block = []
            ms_times_meg_block = []
            cross2_times_meg_block = []
            vs_times_meg_block = []
            buttons_meg_block = []
            response_times_meg_block = []
            response_trials_meg_block = []
            time_diff_block = []
            description_block = []
            onset_block = []

            try:
                # Get events in block from BH data
                bh_evt_times_block = (search_start + rt)[block_idxs]
                responses_block = responses[block_idxs]

                # Realign bh and meg block timelines
                block_time_realign = search_start[block_num * block_trials] + rt[block_num * block_trials] - \
                                     meg_evt_block_times[align_sample]
                bh_evt_times_block = bh_evt_times_block - block_time_realign

                # Iterate over trials
                for trial in range(block_trials):
                    if not np.isnan(bh_evt_times_block[trial]):
                        total_trial = int(block_num * block_trials + trial + 1)

                        idx, meg_evt_time = functions_general.find_nearest(meg_evt_block_times, bh_evt_times_block[trial])
                        onset_block.append(meg_evt_time)
                        description_block.append(meg_evt_block_buttons[idx])

                        time_diff = bh_evt_times_block[trial] - meg_evt_time
                        time_diff_block.append(time_diff)

                        if (meg_evt_block_buttons[idx] == 'blue' and int(responses_block[trial]) != int(
                                subject.buttons_pc_map['blue'])) or (
                                meg_evt_block_buttons[idx] == 'red' and int(responses_block[trial]) != int(
                            subject.buttons_pc_map['red'])):
                            print(f'Different answer in MEG and BH data in trial: {trial + 1}\n'
                                  f'Discarding and realingning on following sample\n')
                            raise ValueError(f'Different answer in MEG and BH data in trial: {trial + 1}')

                        if abs(time_diff) > 0.02:  # and block_num * block_trials + trial not in completed_responses_idx:
                            print(
                                f'{round(abs(meg_evt_time - bh_evt_times_block[trial]) * 1000, 1)} ms difference in Trial: {trial + 1}')

                        # Define screen times from MEG response
                        trial_search_time = meg_evt_time - rt_block_times[trial]
                        vs_times_meg_block.append(trial_search_time)
                        onset_block.append(trial_search_time)
                        description_block.append(f'vs_t{total_trial}')

                        trial_cross2_time = trial_search_time - cross2_block_dur[trial]
                        cross2_times_meg_block.append(trial_cross2_time)
                        onset_block.append(trial_cross2_time)
                        description_block.append(f'cross2_t{total_trial}')

                        trial_ms_time = trial_cross2_time - ms_block_dur[trial]
                        ms_times_meg_block.append(trial_ms_time)
                        onset_block.append(trial_ms_time)
                        description_block.append(f'ms_t{total_trial}')

                        trial_cross1_time = trial_ms_time - cross1_block_dur[trial]
                        cross1_times_meg_block.append(trial_cross1_time)

                        # Save cross1 onset only if there was cross 1 present in that trial
                        if cross1_block_dur[trial]:
                            onset_block.append(trial_cross1_time)
                            description_block.append(f'cross1_t{total_trial}')

                        buttons_meg_block.append((meg_evt_block_buttons[idx]))
                        response_times_meg_block.append(meg_evt_time)
                        response_trials_meg_block.append(int(block_num * block_trials + trial + 1))

                    # Consider manually completed responses
                    # elif np.isnan(bh_evt_times_block[trial]) and responses_block[trial] != 'None':

                    else:
                        print(f'No answer in Trial: {trial + 1}')
                        no_answer_block.append(block_num * block_trials + trial)

                # Define variable of not completed responses times to assess the real time difference
                # (completed responses might artificially increase this value)
                # real_time_diff = [element for trial, element in zip(block_idxs, time_diff_block) if trial not in completed_responses_idx]
                # if np.mean(real_time_diff) > 0.2:
                if np.mean(abs(np.array(time_diff_block))) > 0.2:
                    print(f'Average time difference for this block: {np.mean(abs(np.array(time_diff_block)))} s\n'
                          f'Discarding and realingning on following sample\n')
                    raise ValueError(f'Average time difference for this block over 200 ms')

                else:
                    block_data_aligned = True
                    # Append block data to overall data
                    no_answer.append(no_answer_block)
                    cross1_times_meg.append(cross1_times_meg_block)
                    ms_times_meg.append(ms_times_meg_block)
                    cross2_times_meg.append(cross2_times_meg_block)
                    vs_times_meg.append(vs_times_meg_block)
                    buttons_meg.append(buttons_meg_block)
                    response_times_meg.append(response_times_meg_block)
                    response_trials_meg.append(response_trials_meg_block)
                    time_differences.append(time_diff_block)

                    # Save annotations from block
                    description.append(description_block)
                    onset.append(onset_block)
            except:
                align_sample += 1
                attempts += 1

        if not block_data_aligned:
            raise ValueError(f'Could not align MEG and BH responses in block {block_num + 1}')

    # flatten variables over blocks
    response_trials_meg = functions_general.flatten_list(response_trials_meg)
    cross1_times_meg = functions_general.flatten_list(cross1_times_meg)
    ms_times_meg = functions_general.flatten_list(ms_times_meg)
    cross2_times_meg = functions_general.flatten_list(cross2_times_meg)
    vs_times_meg = functions_general.flatten_list(vs_times_meg)
    buttons_meg = functions_general.flatten_list(buttons_meg)
    response_times_meg = functions_general.flatten_list(response_times_meg)
    time_differences = functions_general.flatten_list(time_differences)
    no_answer = functions_general.flatten_list(no_answer)

    description = functions_general.flatten_list(description)
    onset = functions_general.flatten_list(onset)

    # Save clean events to MEG data
    raw.annotations.description = np.array(description)
    raw.annotations.onset = np.array(onset)

    # Save data to subject class
    subject.trial = np.array(response_trials_meg)
    subject.cross1 = np.array(cross1_times_meg)
    subject.ms = np.array(ms_times_meg)
    subject.cross2 = np.array(cross2_times_meg)
    subject.vs = np.array(vs_times_meg)
    subject.description = np.array(buttons_meg)
    subject.onset = np.array(response_times_meg)
    subject.time_differences = np.array(time_differences)
    subject.no_answer = np.array(no_answer)

    return bh_data, raw, subject


def fixation_classification_BH(subject, bh_data, fixations, raw, meg_pupils_data_clean, exp_info):
    cross1_times_meg = subject.cross1
    response_trials_meg = subject.trial
    ms_times_meg = subject.ms
    cross2_times_meg = subject.cross2
    vs_times_meg = subject.vs
    response_times_meg = subject.onset
    times = raw.times

    # Get mss and target pres/abs from bh data
    mss = bh_data['Nstim'].astype(int)
    pres_abs = bh_data['Tpres'].astype(int)
    corr_ans = bh_data['key_resp.corr'].astype(int)

    if subject.subject_id in exp_info.missing_bh_subjects:
        corr_ans = np.zeros(len(bh_data)).astype(int)
        corr_answers = bh_data['corrAns']
        actual_answers = subject.description
        for i, trial in enumerate(response_trials_meg):
            trial_idx = trial - 1
            if int(subject.buttons_pc_map[actual_answers[i]]) != corr_answers[trial_idx]:  # Check if mapping of button corresponds to correct answer
                corr_ans[trial_idx] = 0
            else:
                corr_ans[trial_idx] = 1

    # Differences only in no answer trials. We're recovering the answers!
    # diff = (corr_ans1-corr_ans).values
    # np.where(diff!=0)[0]+1

    # Get MSS, present/absent for every trial
    fix_trial = []
    fix_screen = []
    trial_mss = []
    tgt_pres_abs = []
    trial_correct = []
    n_fixs = []
    fix_delay = []
    pupil_size = []
    description = []
    onset = []

    # Iterate over fixations to classify them
    print('Classifying fixations')

    # Define dict to store screen-trial fixation number
    fix_numbers = {}
    previous_trial = 0

    for fix_time in fixations['onset'].values:
        # find fixation's trial
        for response_idx, trial_end_time in enumerate(response_times_meg):
            if fix_time < trial_end_time:
                break

        # Define trial to store screen fixation number
        trial = response_trials_meg[response_idx]
        trial_idx = trial - 1
        fix_trial.append(trial)

        if trial != previous_trial:
            fix_numbers[trial] = {'cross1': 0, 'ms': 0, 'cross2': 0, 'vs': 0}

        # First fixation cross
        if cross1_times_meg[response_idx] < fix_time < ms_times_meg[response_idx]:
            screen = 'cross1'
            fix_screen.append(screen)
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - cross1_times_meg[response_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(cross1_times_meg[response_idx] < times, times < ms_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # MS
        elif ms_times_meg[response_idx] < fix_time < cross2_times_meg[response_idx]:
            screen = 'ms'
            fix_screen.append(screen)
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - ms_times_meg[response_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(ms_times_meg[response_idx] < times, times < cross2_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # Second fixations corss
        elif cross2_times_meg[response_idx] < fix_time < vs_times_meg[response_idx]:
            screen = 'cross2'
            fix_screen.append(screen)
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - cross2_times_meg[response_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(cross2_times_meg[response_idx] < times, times < vs_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # VS
        elif vs_times_meg[response_idx] < fix_time < response_times_meg[response_idx]:
            screen = 'vs'
            fix_screen.append(screen)
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - vs_times_meg[response_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(vs_times_meg[response_idx] < times, times < response_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # No screen identified
        else:
            fix_screen.append(None)
            trial_mss.append(None)
            tgt_pres_abs.append(None)
            trial_correct.append(None)
            fix_delay.append(None)
            n_fixs.append(None)
            pupil_size.append(None)

        previous_trial = trial

    fixations['pupil'] = pupil_size
    fixations['trial'] = fix_trial
    fixations['mss'] = trial_mss
    fixations['screen'] = fix_screen
    fixations['target_pres'] = tgt_pres_abs
    fixations['correct'] = trial_correct
    fixations['n_fix'] = n_fixs
    fixations['time'] = fix_delay

    fixations = fixations.astype({'trial': float, 'mss': float, 'target_pres': float, 'time': float, 'n_fix': float, 'pupil': float})

    # Add vs fixations data to raw annotations
    raw.annotations.description = np.concatenate((raw.annotations.description, np.array(description)))
    raw.annotations.onset = np.concatenate((raw.annotations.onset, np.array(functions_general.flatten_list(onset))))

    # Order raw.annotations chronologically
    raw_annot = np.array([raw.annotations.onset, raw.annotations.description])
    raw_annot = raw_annot[:, raw_annot[0].astype(float).argsort()]
    raw.annotations.onset = raw_annot[0].astype(float)
    raw.annotations.description = raw_annot[1]

    # Set durations and ch_names length to match the annotations length
    raw.annotations.duration = np.zeros(len(raw.annotations.description))

    return fixations, raw
