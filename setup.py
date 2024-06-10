import paths
import mne
import pandas as pd
import numpy as np
import pathlib
import os


class exp_info:
    """
    Class containing the experiment information.

    Attributes
    ----------
    ctf_path : str
        Path to the MEG data.
    et_path : str
        Path to the Eye-Tracker data.
    mri_path : str
        Path to the MRI data.
    opt_path : str
        Path to the Digitalization data.
    subjects_ids : list
        List of subject IDs.
    bad_channels : dict
        Dictionary of subject IDs and their corresponding bad channels.
    screen_distance : dict
        Dictionary of subject IDs and their distance to the screen during the experiment.
    screen_size : dict
        Dictionary of subject IDs and their screen size.
    group : dict
        Dictionary of subject IDs and their group (balanced or counterbalanced).
    et_channel_names : dict
        Dictionary of tracked eyes and their corresponding channel names.
    tracked_eye : dict
        Dictionary of subject IDs and their tracked eye.
    no_pupil_subjects : list
        List of subjects with missing pupil data.
    trig_ch : str
        Trigger channel name.
    button_ch : str
        Buttons channel name.
    buttons_ch_map : dict
        Map of button values to colors.
    buttons_pc_map : dict
        Map of button values to colors (PC version).
    DAC_delay : int
        DAC delay in milliseconds.
    noise_recordings : list
        List of background noise recordings date IDs.
    empty_room_recording : dict
        Dictionary of subject IDs and their associated background noise.
    background_line_noise_freqs : dict
        Dictionary of noise recording dates and their line noise frequencies.
    line_noise_freqs : dict
        Dictionary of subject IDs and their line noise frequencies.
    """

    def __init__(self):
        # Define ctf data path and files path
        self.ctf_path = paths.ctf_path
        self.et_path = paths.et_path
        self.mri_path = paths.mri_path
        self.opt_path = paths.opt_path

        # Select subject
        self.subjects_ids = ['16991001', '15584005', '13229005', '13703055', '16589013', '16425014', '17439002', '17438002', '17647001', '17478002', '16746003', '17634001']

        # Subjects bad channels
        self.bad_channels = {'16991001': [],
                             '15584005': [],
                             '13229005': [],
                             '13703055': [],
                             '16589013': [],
                             '16425014': [],
                             '17439002': [],
                             '17438002': [],
                             '17647001': [],
                             '17478002': [],
                             '16746003': [],
                             '17634001': []
                             }

        # Distance to the screen during the experiment
        # It's actually camera Check with matias if this is what we want (top/bottop of the screen)
        self.screen_distance = {'16991001': 54.5,
                                '15584005': 56.5,
                                '13229005': 59.5,
                                '13703055': 59.5,
                                '16589013': 58,
                                '16425014': 58,
                                '17439002': 56,
                                '17438002': 48,
                                '17647001': 55.2,
                                '17478002': 57,
                                '16746003': 52.5,
                                '17634001': 59.5}

        # Screen width
        self.screen_size = {'16991001': 37.5,
                            '15584005': 37.5,
                            '13229005': 38,
                            '13703055': 37.5,
                            '16589013': 38,
                            '16425014': 38,
                            '17439002': 37.5,
                            '17438002': 34.5,
                            '17647001': 34,
                            '17478002': 34,
                            '16746003': 34,
                            '17634001': 34}

        # Subjects groups
        # For some reason participant 6 has the mapping from balanced participants
        self.group = {'16991001': 'balanced',
                      '15584005': 'counterbalanced',
                      '13229005': 'balanced',
                      '13703055': 'counterbalanced',
                      '16589013': 'balanced',
                      '16425014': 'balanced',
                      '17439002': 'counterbalanced',
                      '17438002': 'balanced',
                      '17647001': 'counterbalanced',
                      '17478002': 'balanced',
                      '16746003': 'counterbalanced',
                      '17634001': 'balanced'}

        # Tracked eye
        self.tracked_eye = {'16991001': 'right',
                            '15584005': 'both',
                            '13229005': 'both',
                            '13703055': 'both',
                            '16589013': '?',
                            '16425014': '?',
                            '17439002': 'right',
                            '17438002': 'left',
                            '17647001': 'left',
                            '17478002': 'both',
                            '16746003': 'both',
                            '17634001': 'left'}

        # Get et channels by name [Gaze x, Gaze y, Pupils]
        self.et_channel_names = {'16991001': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '15584005': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '13229005': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '13703055': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '16589013': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '16425014': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17439002': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17438002': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17647001': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17478002': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '16746003': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17634001': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123']}

        # Missing pupil data
        self.no_pupil_subjects = []

        # Trigger channel name
        self.trig_ch = 'UPPT002'

        # Buttons channel name
        self.button_ch = 'UPPT001'

        # Buttons values to colors map
        self.buttons_ch_map = {1: 'blue', 2: 'yellow', 4: 'green', 8: 'red', 'blue': 1, 'yellow': 2, 'green': 4, 'red': 8}
        self.buttons_pc_map = {1: 'blue', 2: 'yellow', 3: 'green', 4: 'red', 'blue': 1, 'yellow': 2, 'green': 3, 'red': 4}

        # DAC delay (in ms)
        self.DAC_delay = 10

        # Background noise recordings date ids
        self.noise_recordings = ['20230630', '20230705', '20230801', '20230803', '20240305', '20240327', '20240408', '20240507']

        # Participants associated background noise
        self.empty_room_recording = {'16991001': '20230630',
                                     '15584005': '20230705',
                                     '13229005': '20230801',
                                     '13703055': '20230803',
                                     '16589013': '20240305',
                                     '16425014': '20240327',
                                     '17439002': '20240408',
                                     '17438002': '20240503',
                                     '17647001': '20240507',
                                     '17478002': '20240510',
                                     '16746003': '20240513',
                                     '17634001': '20240514'}

        # Notch filter line noise frequencies
        self.background_line_noise_freqs = {'20230630': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20230705': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20230801': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20230803': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240305': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240327': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240408': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240507': (50, 57, 100, 109, 150, 200, 250, 300),
                                            }

        # Notch filter line noise frequencies
        self.line_noise_freqs = {'16991001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '15584005': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '13229005': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '13703055': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16589013': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16425014': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17439002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17438002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17647001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17478002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16746003': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17634001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 }


class analysis_parameters:
    """
    Class containing the analysis parameters.

    Attributes
    ----------
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subjects_ids: list
        List of subject's id.
    subjects_bad_channels: list
        List of subject's bad channels.
    subjects_groups: list
        List of subject's group
    missing_bh_subjects: list
        List of subject's ids missing behavioural data.
    trials_loop_subjects: list
        List of subject;s ids for subjects that took the firts version of the experiment.
    """

    def __init__(self):
        self.preprocessing = self.preprocessing()
        self.general = self.general()

    class preprocessing:
        def __init__(self):
            # Samples drop at begining of missing pupils signal
            self.start_interval_samples = 24

            # Samples drop at end of missing pupils signal
            self.end_interval_samples = 24

            # Pupil size threshold to consider missing signal
            self.pupil_thresh = {'16991001': -4.5,
                                 '15584005': -4.6,
                                 '13229005': -4.6,
                                 '13703055': -4.6,
                                 '16589013': -4.6,
                                 '16425014': -4.6,
                                 '17439002': -4.6,
                                 '17438002': -4.6,
                                 '17647001': -4.6,
                                 '17478002': -4.6,
                                 '16746003': -4.6,
                                 '17634001': -4.6}

            # Et samples shift for ET-MEG alignment
            self.et_samples_shift = {}

    class general:
        def __init__(self):
            # Trial reject parameter based on MEG peak to peak amplitude
            self.reject_amp = {'16991001': 5e-12,
                               '15584005': 5e-12,
                               '13229005': 5e-12,
                               '13703055': 5e-12,
                               '16589013': 5e-12,
                               '16425014': 5e-12,
                               '17439002': 5e-12,
                               '17438002': 5e-12,
                               '17647001': 5e-12,
                               '17478002': 5e-12,
                               '16746003': 5e-12,
                               '17634001': 5e-12}

            # Subjects dev <-> head transformation to use (which head localization)
            self.head_loc_idx = {'16991001': 0,
                                 '15584005': 0,
                                 '13229005': 0,
                                 '13703055': 0,
                                 '16589013': 0,
                                 '16425014': 0,
                                 '17439002': 0,
                                 '17438002': 0,
                                 '17647001': 0,
                                 '17478002': 0,
                                 '16746003': 0,
                                 '17634001': 0}


class raw_subject():
    """
    Class containing subjects data and analysis parameters.

    Parameters
    ----------
    subject: {'int', 'str'}, default=None
        Subject id (str) or number (int). If None, takes the first subject.

    Attributes
    ----------
    bad_channels: list
        List of bad channels.
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subject_id: str
        Subject id.
    """

    def __init__(self, subject_code=None):

        # Select 1st subject by default
        if subject_code == None:
            self.subject_id = exp_info().subjects_ids[0]
        # Select subject by index
        elif type(subject_code) == int:
            self.subject_id = exp_info().subjects_ids[subject_code]
        # Select subject by id
        elif type(subject_code) == str and (subject_code in exp_info().subjects_ids):
            self.subject_id = subject_code
        else:
            print('Subject not found')

        # Get attributes from experiment info
        exp_info_att = exp_info().__dict__.keys()
        for general_att in exp_info_att:
            att = getattr(exp_info(), general_att)
            if type(att) == dict:
                try:
                    # If subject_id in dictionary keys, get attribute
                    att_value = att[self.subject_id]
                    setattr(self, general_att, att_value)
                except:
                    pass

        # Get preprocessing and general configurations
        self.params = self.params(params=analysis_parameters(), subject_id=self.subject_id)

    # Subject's parameters and configuration
    class params:

        def __init__(self, params, subject_id):
            self.preproc = self.preproc(params=params, subject_id=subject_id)
            self.general = self.general(params=params, subject_id=subject_id)

        # Configuration for preprocessing run
        class preproc:
            def __init__(self, params, subject_id):

                # Get config.preprocessing attributes and get data for corresponding subject
                preproc_attributes = params.preprocessing.__dict__.keys()

                # Iterate over attributes and get data for corresponding subject
                for preproc_att in preproc_attributes:
                    att = getattr(params.preprocessing, preproc_att)
                    if type(att) == dict:
                        try:
                            # If subject_id in dictionary keys, get attribute, else pass
                            att_value = att[subject_id]
                            setattr(self, preproc_att, att_value)
                        except:
                            pass
                    else:
                        # If attribute is general for all subjects, get attribute
                        att_value = att
                        setattr(self, preproc_att, att_value)

        # Configuration for further analysis
        class general:
            def __init__(self, params, subject_id):

                # Get config.preprocessing attirbutes and get data for corresponding subject
                general_attributes = params.general.__dict__.keys()

                # Iterate over attributes and get data for conrresponding subject
                for general_att in general_attributes:
                    att = getattr(params.general, general_att)
                    if type(att) == dict:
                        try:
                            # If subject_id in dictionary keys, get attribute, else pass
                            att_value = att[subject_id]
                            setattr(self, general_att, att_value)
                        except:
                            pass
                    else:
                        # If attribute is general for all subjects, get attribute
                        att_value = att
                        setattr(self, general_att, att_value)

    # Raw MEG data
    def load_raw_meg_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading Raw MEG data')
        # Get subject path
        subj_path = pathlib.Path(os.path.join(paths.ctf_path, self.subject_id))
        ds_files = list(subj_path.glob('*{}*.ds'.format(self.subject_id)))
        ds_files.sort()

        # Load sesions
        # If more than 1 session concatenate all data to one raw data
        if len(ds_files) > 1:
            raws_list = []
            for i in range(len(ds_files)):
                raw = mne.io.read_raw_ctf(ds_files[i], system_clock='ignore')
                raws_list.append(raw)
            # MEG data structure
            raw = mne.io.concatenate_raws(raws_list, on_mismatch='ignore')

            # Set dev <-> head transformation from optimal head localization
            raw.info['dev_head_t'] = raws_list[self.params.general.head_loc_idx].info['dev_head_t']

        # If only one session return that session as whole raw data
        elif len(ds_files) == 1:
            raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')

        # Missing data
        else:
            raise ValueError('No .ds files found in subject directory: {}'.format(subj_path))

        return raw

    # MEG data
    def load_preproc_meg_data(self, preload=False):
        """
        Preprocessed MEG data for parent subject as raw instance of MNE.
        """

        # Subject preprocessed data path
        file_path = pathlib.Path(os.path.join(paths.preproc_path, self.subject_id, f'Subject_{self.subject_id}_meg.fif'))

        # Try to load preprocessed data
        try:
            print('\nLoading Preprocessed MEG data')
            meg_data = mne.io.read_raw_fif(file_path, preload=preload)
        except:
            raise ValueError(f'No previous preprocessed data found for subject {self.subject_id}')

        return meg_data

    # ICA MEG data
    def load_ica_meg_data(self, preload=False):
        """
        ICA MEG data for parent subject as Raw instance of MNE.
        """

        # Subject ICA data path
        file_path = pathlib.Path(os.path.join(paths.ica_path, self.subject_id, f'Subject_{self.subject_id}_ICA.fif'))

        # Try to load ica data
        try:
            print(f'Loading ICA data for subject {self.subject_id}')
            # Load data
            ica_data = mne.io.read_raw_fif(file_path, preload=preload)

        except:
            raise ValueError(f'No previous ica data found for subject {self.subject_id}')

        return ica_data

    # ET data
    def load_raw_et_data(self):
        """
        Eye-Tracker data for parent subject as dict containing pandas DataFrames.

        Attributes
        -------
        asc: DataFrame
            Entire asc file data.
        start_time: str
            Recording start time in ms.
        samples_start: int
            Line number in asc file where ET samples start.
        head: DataFrame
            asc file header.
        eye: str
            Tracked eye.
        samples: DataFrame
            ET data. Columns: Time, Gaze x, Gaze y, Pupil size
        time: Series
            Data time series. Corresponding to the first columns of the samples DataFrame.
        fix: DataFrame
            Fixations.
        sac: DataFrame:
            Saccades.
        blinks: DataFrame
            Blinks.
        sync: DataFrame
            Synchronization messages.
        msg: DataFrame
            All Messages recieved by the Eye-Tracker.
        calibration: Series
            Calibration messages. First value indicates time of message recieved.
        """

        print('\nLoading ET data')
        # get subject path
        subj_path = pathlib.Path(os.path.join(paths.et_path, self.subject_id))
        # Load asc file
        asc_file_path = list(subj_path.glob('*{}*.asc'.format(self.subject_id)))[0]

        # data structure
        et = {}

        # ASC FILE
        print('Reading asc')
        et['asc'] = pd.read_table(asc_file_path, names=np.arange(9), low_memory=False)

        # INFO
        et['start_time'] = et['asc'][1][np.where(et['asc'][0] == 'START')[0][0]]
        et['samples_start'] = np.where(et['asc'][0] == et['start_time'].split()[0])[0][0]
        et['head'] = et['asc'].iloc[:et['samples_start']]
        et['eye'] = et['head'].loc[et['head'][0] == 'EVENTS'][2].values[0]
        print('Loading headers: {}'.format(et['samples_start']))

        # auxiliar numeric df
        print('Converting asc to numeric')
        num_et = et['asc'].apply(pd.to_numeric, errors='coerce')

        # SAMPLES
        # et['samples'] = num_et.loc[~pd.isna(num_et[np.arange(4)]).any(1)][np.arange(4)]
        et['samples'] = num_et.loc[~pd.isna(num_et[0])][np.arange(4)]
        print('Loading samples: {}'.format(len(et['samples'])))

        # TIME
        et['time'] = num_et[0].loc[~pd.isna(num_et[0])]
        # et['time'] = num_et[0]
        print('Loading time: {}'.format(len(et['time'])))

        # FIXATIONS
        et['fix'] = et['asc'].loc[et['asc'][0].str.contains('EFIX').values == True][np.arange(5)]
        et['fix'][0] = et['fix'][0].str.split().str[-1]
        et['fix'] = et['fix'].apply(pd.to_numeric, errors='coerce')
        print('Loading fixations: {}'.format(len(et['fix'])))

        # SACADES
        et['sac'] = et['asc'].loc[et['asc'][0].str.contains('ESAC').values == True][np.arange(9)]
        et['sac'][0] = et['sac'][0].str.split().str[-1]
        et['sac'] = et['sac'].apply(pd.to_numeric, errors='coerce')
        print('Loading saccades: {}'.format(len(et['sac'])))

        # BLINKS
        et['blinks'] = et['asc'].loc[et['asc'][0].str.contains('EBLINK').values == True][np.arange(3)]
        et['blinks'][0] = et['blinks'][0].str.split().str[-1]
        et['blinks'] = et['blinks'].apply(pd.to_numeric, errors='coerce')
        print('Loading blinks: {}'.format(len(et['blinks'])))

        # ETSYNC
        et['sync'] = et['asc'].loc[et['asc'][1].str.contains('ETSYNC').values == True][np.arange(3)]
        et['sync'][0] = et['sync'][1].str.split().str[1]
        et['sync'][2] = pd.to_numeric(et['sync'][1].str.split().str[2])
        et['sync'][1] = pd.to_numeric(et['sync'][1].str.split().str[0])
        print('Loading sync messages: {}'.format(len(et['sync'])))

        # MESSAGES
        et['msg'] = et['asc'].loc[et['asc'][0].str.contains('MSG').values == True][np.arange(2)]
        print('Loading all messages: {}'.format(len(et['msg'])))

        # CALIBRATION
        et['calibration'] = et['asc'].loc[et['asc'][0].str.contains('CALIBRATION').values == True][0]
        print('Loading calibration messages: {}'.format(len(et['calibration'])))

        return et


    # Behavioural data
    def load_raw_bh_data(self):
        """
        Behavioural data for parent subject as pandas DataFrames.
        """
        # Get subject path
        raise ('Update paths to use csv within ET data.')
        subj_path = pathlib.Path(os.path.join(paths.eh_path, self.subject_id))
        bh_file = list(subj_path.glob('*.csv'.format(self.subject_id)))[0]

        # Load DataFrame
        df = pd.read_csv(bh_file)

        return df


class noise:
    """
    Class containing bakground noise data.

    Parameters
    ----------
    exp_info:
    config:
    subject: {'int', 'str'}, default=None
        Subject id (str) or number (int). If None, takes the first subject.

    Attributes
    ----------
    bad_channels: list
        List of bad channels.
    beh_path: str
        Path to the behavioural data.
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subject_id: str
        Subject id.
    """

    def __init__(self, date_id):

        # Noise recording date id
        self.subject_id = date_id

        # Back Noise directory name
        self.bkg_noise_dir = 'BACK_NOISE'

        # Noise data path
        self.ctf_path = pathlib.Path(os.path.join(exp_info().ctf_path, self.bkg_noise_dir))

    # MEG data
    def load_raw_meg_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading MEG data')
        # get subject path
        subj_path = pathlib.Path(os.path.join(paths.ctf_path, 'BACK_NOISE'))
        ds_files = list(subj_path.glob('QA_*_{}_01.ds'.format(self.subject_id)))
        ds_files.sort()

        # Load sesions
        # If more than 1 session concatenate all data to one raw data
        if len(ds_files) > 1:
            raws_list = []
            for i in range(len(ds_files)):
                raw = mne.io.read_raw_ctf(ds_files[i], system_clock='ignore')
                raws_list.append(raw)
            # MEG data structure
            raw = mne.io.concatenate_raws(raws_list, on_mismatch='ignore')

        # If only one session return that session as whole raw data
        elif len(ds_files) == 1:
            raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')

        # Missing data
        else:
            raise ValueError(f'No {ds_files} files found in subject directory: {subj_path}')

        return raw

    def load_preproc_data(self, preload=False):
        """
        Preprocessed MEG data for parent subject as raw instance of MNE.
        """

        print('\nLoading Preprocessed MEG data')
        # get subject path
        preproc_path = paths.preproc_path
        file_path = pathlib.Path(os.path.join(preproc_path, self.bkg_noise_dir, f'{self.subject_id}_meg.fif'))

        # Load data
        fif = mne.io.read_raw_fif(file_path, preload=preload)

        return fif


class all_subjects:

    def __init__(self, all_fixations, all_saccades, all_bh_data, all_rt, all_corr_ans, all_mss):
        self.subject_id = 'All_Subjects'
        self.fixations = all_fixations
        self.saccades = all_saccades
        self.trial = np.arange(1, 211)
        self.bh_data = all_bh_data
        self.rt = all_rt
        self.corr_ans = all_corr_ans
        self.mss = all_mss
