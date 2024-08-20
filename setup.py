import paths
import mne
import pandas as pd
import numpy as np
import pathlib
import os
import functions_preproc

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
        self.subjects_ids = ['16991001',
                             '15584005',
                             '13229005',
                             '13703055',
                             '16589013',
                             '16425014',
                             '17439002',
                             '17438002',
                             '17647001',
                             '17478002',
                             '6398006',
                             '16746003',
                             '17634001',
                             '17660001',
                             '17700001',
                             '17734001',
                             '16432016',
                             '15435012',
                             '17772001',
                             '17773002',
                             '17781001',
                             '11766061',
                             '17782001',
                             '17735002',
                             '17788002',
                             ]

        # Subjects bad channels
        self.bad_channels = {'16746003': [],
                             '17634001': [],
                             '17660001': ['MLT52'],
                             '17700001': ['MLO42'],
                             '17734001': ['MLT47', 'MLT57', 'MLO14'],
                             '16432016': [],
                             '15435012': [],
                             '17772001': [],
                             '17773002': ['MRF11'],
                             '17781001': [],
                             '11766061': [],
                             '17782001': [],
                             '17735002': [],
                             '17788002': ['MLT47'],
                             '16991001': [],
                             '15584005': [],
                             '13229005': ['MRT52', 'MRT43'],
                             '13703055': [],
                             '16589013': [],
                             '16425014': [],
                             '17439002': [],
                             '17438002': ['MLT47'],
                             '17647001': ['MLT47', 'MLT57'],
                             '17478002': [],
                             '6398006': []
                             }

        # Distance to the screen during the experiment
        # It's actually camera Check with matias if this is what we want (top/bottop of the screen)
        self.screen_distance = {'16746003': 65.5,
                                '17634001': 71.5,
                                '17660001': 63,
                                '17700001': 68,
                                '17734001': 64,
                                '16432016': 69.2,
                                '15435012': 69,
                                '17772001': 68,
                                '17773002': 66,
                                '17781001': 67,
                                '11766061': 67.5,
                                '17782001': 67,
                                '17735002': 68,
                                '16991001': 65.5,
                                '15584005': 68,
                                '13229005': 71.5,
                                '13703055': 71.5,
                                '16589013': 68,
                                '16425014': 68,
                                '17439002': 68,
                                '17438002': 57.7,
                                '17647001': 66.4,
                                '17478002': 68,
                                '6398006': 68
                                }

        # Screen width
        self.screen_size = {'16746003': 34,
                            '17634001': 34,
                            '17660001': 35,
                            '17700001': 34,
                            '17734001': 34,
                            '16432016': 34,
                            '15435012': 34,
                            '17772001': 34,
                            '17773002': 34,
                            '17781001': 34,
                            '11766061': 34,
                            '17782001': 34,
                            '17735002': 33,
                            '17788002': 33,
                            '16991001': 37.5,
                            '15584005': 37.5,
                            '13229005': 38,
                            '13703055': 37.5,
                            '16589013': 37.5,
                            '16425014': 37.5,
                            '17439002': 37.5,
                            '17438002': 34.5,
                            '17647001': 34,
                            '17478002': 34,
                            '6398006': 34
                            }

        # Subjects groups
        # For some reason participant 6 has the mapping from balanced participants
        self.group = {'16746003': 'counterbalanced',
                      '17634001': 'balanced',
                      '17660001': 'counterbalanced',
                      '17700001': 'balanced',
                      '17734001': 'counterbalanced',
                      '16432016': 'balanced',
                      '15435012': 'counterbalanced',
                      '17772001': 'balanced',
                      '17773002': 'balanced',
                      '17781001': 'counterbalanced',
                      '11766061': 'balanced',
                      '17782001': 'counterbalanced',
                      '17735002': 'counterbalanced',
                      '17788002': 'balanced',
                      '16991001': 'balanced',
                      '15584005': 'counterbalanced',
                      '13229005': 'balanced',
                      '13703055': 'counterbalanced',
                      '16589013': 'balanced',
                      '16425014': 'balanced',
                      '17439002': 'counterbalanced',
                      '17438002': 'balanced',
                      '17647001': 'counterbalanced',
                      '17478002': 'balanced',
                      '6398006': 'counterbalanced'
                      }

        # Tracked eye
        self.tracked_eye = {'16746003': 'left',
                            '17634001': 'left',
                            '17660001': 'right',
                            '17700001': 'right',
                            '17734001': 'left',
                            '16432016': 'left',
                            '15435012': 'right',
                            '17772001': 'left',
                            '17773002': 'left',
                            '17781001': 'right',
                            '11766061': 'right',
                            '17782001': 'left',
                            '17735002': 'right',
                            '17788002': 'left',
                            '16991001': 'right',
                            '15584005': 'right',
                            '13229005': 'left',
                            '13703055': 'left',
                            '16589013': 'left',
                            '16425014': 'right',
                            '17439002': 'right',
                            '17438002': 'right',
                            '17647001': 'right',
                            '17478002': 'right',
                            '6398006': 'left'
                            }

        # Get et channels by name [Gaze x, Gaze y, Pupils]
        self.et_channel_names = {'16746003': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '17634001': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '17660001': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17700001': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17734001': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '16432016': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '15435012': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17772001': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '17773002': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '17781001': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '11766061': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17782001': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '17735002': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17788002': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '16991001': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '15584005': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '13229005': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '13703055': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '16589013': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '16425014': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17439002': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17438002': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17647001': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '17478002': ['UADC001-4123', 'UADC002-4123', 'UADC014-4123'],
                                 '6398006': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123']
                                 }

        # Missing pupil data
        self.no_pupil_subjects = []

        # Alternative trigger channel
        self.alt_trigger = ['17734001', '16432016']

        # Trigger channel name
        self.trig_ch = 'UPPT002'
        self.alt_trig_ch = 'UPPT001'

        # Buttons channel name
        self.button_ch = 'UPPT001'

        # Buttons values to colors map
        self.buttons_ch_map = {1: 'blue', 2: 'yellow', 4: 'green', 8: 'red', 'blue': 1, 'yellow': 2, 'green': 4, 'red': 8}
        self.buttons_pc_map = {1: 'blue', 2: 'yellow', 3: 'green', 4: 'red', 'blue': 1, 'yellow': 2, 'green': 3, 'red': 4}

        # DAC delay (in ms)
        self.DAC_delay = 10

        # Background noise recordings date ids
        self.noise_recordings = ['20240513', '20240514', '20240604', '20240606', '20240612', '20240613', '20240626', '20240701', '20240705', '20240702',
                                 '20240617', '20240709', '20230630', '20230705', '20230801', '20230803', '20240305', '20240327', '20240408', '20240507']

        # Participants associated background noise
        self.empty_room_recording = {'16746003': '20240513',
                                     '17634001': '20240514',
                                     '17660001': '20240604',
                                     '17700001': '20240606',
                                     '17734001': '20240612',
                                     '16432016': '20240613',
                                     '15435012': '20240613',
                                     '17772001': '20240626',
                                     '17773002': '20240701',
                                     '17781001': '20240705',
                                     '11766061': '20240702',
                                     '17782001': '20240709',
                                     '17735002': '20240617',
                                     '17788002': '20240709',
                                     '16991001': '20230630',
                                     '15584005': '20230705',
                                     '13229005': '20230801',
                                     '13703055': '20230803',
                                     '16589013': '20240305',
                                     '16425014': '20240327',
                                     '17439002': '20240408',
                                     '17438002': '20240503',
                                     '17647001': '20240507',
                                     '17478002': '20240510',
                                     '6398006': '20240626'
                                     }

        # Notch filter line noise frequencies
        self.background_line_noise_freqs = {'20240513': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240514': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20230801': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20230803': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240305': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240327': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240408': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240617': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240507': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20230630': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20230705': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20240626': (50, 57, 100, 109, 150, 200, 250, 300)
                                            }

        # Notch filter line noise frequencies
        self.line_noise_freqs = {'16746003': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17634001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17660001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17700001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17734001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16432016': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '15435012': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17772001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17773002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17781001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '11766061': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17782001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17735002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17788002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16991001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '15584005': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '13229005': (57, 75, 109, 150, 200, 225, 250, 300),
                                 '13703055': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16589013': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16425014': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17439002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17438002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17647001': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '17478002': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '6398006': (50, 57, 100, 109, 150, 200, 250, 300),
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
            self.pupil_thresh = {'16746003': -4,
                                 '17634001': -3.8,
                                 '17660001': -4.6,
                                 '17700001': -4.5,
                                 '17734001': -3.6,
                                 '16432016': -4.1,
                                 '15435012': -3.7,
                                 '17772001': -4.6,
                                 '17773002': -4.2,
                                 '17781001': -4.6,
                                 '11766061': -4.2,
                                 '17782001': -4.2,
                                 '17735002': -4.1,
                                 '17788002': -4.7,
                                 '16991001': -4.4,
                                 '15584005': -4.6,
                                 '13229005': -3.6,
                                 '13703055': -4.8,
                                 '16589013': -4.6,
                                 '16425014': -4.9,
                                 '17439002': -4.7,
                                 '17438002': -4,
                                 '17647001': -4.5,
                                 '17478002': -4,
                                 '6398006': -2.6
                                 }

            # Et samples shift for ET-MEG alignment
            self.et_samples_shift = {}

    class general:
        def __init__(self):
            # Trial reject parameter based on MEG peak to peak amplitude
            self.reject_amp = {'16746003': 5e-12,
                               '17634001': 5e-12,
                               '17660001': 5e-12,
                               '17700001': 5e-12,
                               '17734001': 5e-12,
                               '16432016': 5e-12,
                               '15435012': 5e-12,
                               '17772001': 5e-12,
                               '17773002': 5e-12,
                               '17781001': 5e-12,
                               '11766061': 5e-12,
                               '17782001': 5e-12,
                               '17735002': 5e-12,
                               '17788002': 5e-12,
                               '16991001': 5e-12,
                               '15584005': 5e-12,
                               '13229005': 5e-12,
                               '13703055': 5e-12,
                               '16589013': 5e-12,
                               '16425014': 5e-12,
                               '17439002': 5e-12,
                               '17438002': 5e-12,
                               '17647001': 5e-12,
                               '17478002': 5e-12,
                               '6398006': 5e-12
                               }

            # Subjects dev <-> head transformation to use (which head localization)
            self.head_loc_idx = {'16746003': 0,
                                 '17634001': 0,
                                 '17660001': 0,
                                 '17700001': 0,
                                 '17734001': 0,
                                 '16432016': 0,
                                 '15435012': 0,
                                 '17772001': 0,
                                 '17773002': 0,
                                 '17781001': 0,
                                 '11766061': 0,
                                 '17782001': 0,
                                 '17735002': 0,
                                 '17788002': 0,
                                 '16991001': 0,
                                 '15584005': 0,
                                 '13229005': 0,
                                 '13703055': 0,
                                 '16589013': 0,
                                 '16425014': 0,
                                 '17439002': 0,
                                 '17438002': 0,
                                 '17647001': 0,
                                 '17478002': 0,
                                 '6398006': 0
                                 }


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
        edf_file_path = str(pathlib.Path(os.path.join(subj_path, f'{self.subject_id}.edf')))
        asc_file_path = functions_preproc.convert_edf_to_ascii(edf_file_path=edf_file_path, output_dir=subj_path)

        # ===== READ IN FILES ===== #
        # Read in EyeLink file
        f = open(asc_file_path, 'r')
        fileTxt0 = f.read().splitlines(True)  # split into lines
        fileTxt0 = list(filter(None, fileTxt0))  # remove emptys
        fileTxt0 = np.array(fileTxt0)  # concert to np array for simpler indexing
        f.close()

        # Separate lines into samples and messages
        print('Sorting lines...')
        nLines = len(fileTxt0)
        lineType = np.array(['OTHER'] * nLines, dtype='object')

        # Usar lo de mne, particularmente para calibration.
        # En sample tendría que filtrar lo que viene después de START y antes de END.

        calibration_flag = False
        start_flag = False
        for iLine in range(nLines):
            if len(fileTxt0[iLine]) < 3:
                lineType[iLine] = 'EMPTY'
            elif fileTxt0[iLine].startswith('*'):
                lineType[iLine] = 'HEADER'
            # If there is a !CAL in the line, it is a calibration line
            elif '!CAL' in fileTxt0[iLine]:
                lineType[iLine] = 'Calibration'
                calibration_flag = True
            elif fileTxt0[iLine].split()[0] == 'START' and calibration_flag:
                calibration_flag = False
                start_flag = True
            elif calibration_flag:
                lineType[iLine] = 'Calibration'
            elif not start_flag:  # Data before the first calibration is discarded
                lineType[iLine] = 'Non_calibrated_samples'
            elif fileTxt0[iLine].split()[0] == 'MSG':
                lineType[iLine] = 'MSG'
            elif fileTxt0[iLine].split()[0] == 'ESACC':
                lineType[iLine] = 'ESACC'
            elif fileTxt0[iLine].split()[0] == 'EFIX':
                lineType[iLine] = 'EFIX'
            elif fileTxt0[iLine].split()[0] == 'EBLINK':
                lineType[iLine] = 'EBLINK'
            elif fileTxt0[iLine].split()[0][0].isdigit() or fileTxt0[iLine].split()[0].startswith('-'):
                lineType[iLine] = 'SAMPLE'
            else:
                lineType[iLine] = 'OTHER'

        # ===== PARSE EYELINK FILE ===== #
        # Import Header
        print('Parsing header...')
        dfHeader = pd.read_csv(asc_file_path, skiprows=np.nonzero(lineType != 'HEADER')[0], header=None, sep='\s+')
        # Merge columns into single strings
        dfHeader = dfHeader.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

        # Import Calibration
        print('Parsing calibration...')
        iCal = np.nonzero(lineType != 'Calibration')[0]
        dfCalib = pd.read_csv(asc_file_path, skiprows=iCal, names=np.arange(9))

        # Import Message
        print('Parsing messages...')
        i_msg = np.nonzero(lineType == 'MSG')[0]
        t_msg = []
        txt_msg = []
        for i in range(len(i_msg)):
            # separate MSG prefix and timestamp from rest of message
            info = fileTxt0[i_msg[i]].split()
            # extract info
            t_msg.append(int(info[1]))
            txt_msg.append(' '.join(info[2:]))
        dfMsg = pd.DataFrame({'time': t_msg, 'text': txt_msg})

        # Import Fixations
        print('Parsing fixations...')
        i_not_efix = np.nonzero(lineType != 'EFIX')[0]
        df_fix = pd.read_csv(asc_file_path, skiprows=i_not_efix, header=None, sep='\s+', usecols=range(1, 8), low_memory=False)
        df_fix.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg', 'pupilAvg']

        # Saccades
        print('Parsing saccades...')
        i_not_esacc = np.nonzero(lineType != 'ESACC')[0]
        df_sacc = pd.read_csv(asc_file_path, skiprows=i_not_esacc, header=None, sep='\s+', usecols=range(1, 11), low_memory=False)
        df_sacc.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak']

        # Blinks
        print('Parsing blinks...')
        df_blink = pd.DataFrame()
        i_not_eblink = np.nonzero(lineType != 'EBLINK')[0]
        if len(i_not_eblink) < nLines:
            df_blink = pd.read_csv(asc_file_path, skiprows=i_not_eblink, header=None, sep='\s+', usecols=range(1, 5), low_memory=False)
            df_blink.columns = ['eye', 'tStart', 'tEnd', 'duration']

        # determine sample columns based on eyes recorded in file
        eyes_in_file = np.unique(df_fix.eye)
        if eyes_in_file.size == 2:
            cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
        else:
            eye = eyes_in_file[0]
            print('monocular data detected (%c eye).' % eye)
            cols = ['tSample', '%cX' % eye, '%cY' % eye, '%cPupil' % eye]

        # Import samples
        i_not_sample = np.nonzero(lineType != 'SAMPLE')[0]
        dfSamples = pd.read_csv(asc_file_path, skiprows=i_not_sample, header=None, sep='\s+', usecols=range(0, len(cols)), low_memory=False)
        dfSamples.columns = cols
        # Convert values to numbers
        for eye in ['L', 'R']:
            if eye in eyes_in_file:
                dfSamples['%cX' % eye] = pd.to_numeric(dfSamples['%cX' % eye], errors='coerce')
                dfSamples['%cY' % eye] = pd.to_numeric(dfSamples['%cY' % eye], errors='coerce')
                dfSamples['%cPupil' % eye] = pd.to_numeric(dfSamples['%cPupil' % eye], errors='coerce')
            else:
                dfSamples['%cX' % eye] = np.nan
                dfSamples['%cY' % eye] = np.nan
                dfSamples['%cPupil' % eye] = np.nan

        dict_events = {'fix': df_fix, 'sacc': df_sacc, 'blink': df_blink}

        # Return dictionary
        et = {}
        et['samples'] =dfSamples
        et['cal'] = dfCalib
        et['header'] = dfHeader
        et['msg'] = dfMsg
        et['events'] = dict_events

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
