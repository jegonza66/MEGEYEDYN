import os.path
import functions_general
import plot_general
import functions_analysis
import setup
import save
import plot_preproc
import functions_preproc
import paths
import mne
import matplotlib.pyplot as plt

# Load configuration
exp_info = setup.exp_info()

# Run
for subject_code in exp_info.subjects_ids:

    # ---------------- Load data ----------------#
    # Define subject
    subject = setup.raw_subject(subject_code=subject_code)

    # Load Meg data
    raw = subject.load_raw_meg_data()

    # Get ET channels from MEG
    print('\nGetting ET channels data from MEG')
    et_channels_meg = raw.get_data(picks=exp_info.et_channel_names[subject.tracked_eye])

    # ---------------- Remove DAC delay samples ----------------#
    meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw = functions_preproc.DAC_samples(et_channels_meg=et_channels_meg,
                                                                                                exp_info=exp_info,
                                                                                                sfreq=raw.info['sfreq'])

    # ---------------- Reescaling based on conversion parameters ----------------#
    meg_gazex_data_scaled, meg_gazey_data_scaled = functions_preproc.reescale_et_channels(meg_gazex_data_raw=meg_gazex_data_raw,
                                                                                          meg_gazey_data_raw=meg_gazey_data_raw)

    # ---------------- Blinks removal ----------------#
    # Define intervals around blinks to also fill with nan. Due to conversion noise from square signal
    et_channels_meg = functions_preproc.blinks_to_nan(exp_info=exp_info,
                                                      subject=subject,
                                                      meg_gazex_data_scaled=meg_gazex_data_scaled,
                                                      meg_gazey_data_scaled=meg_gazey_data_scaled,
                                                      meg_pupils_data_raw=meg_pupils_data_raw)

    # ---------------- Add scaled ET data to MEG data as new channels ----------------#
    raw = functions_preproc.add_et_channels(raw=raw,
                                            et_channels_meg=et_channels_meg,
                                            et_channel_names=subject.et_channel_names)

    # ---------------- Filter line noise ----------------#
    filtered_data = functions_preproc.filter_line_noise(subject=subject,
                                                        raw=raw,
                                                        freqs=subject.line_noise_freqs)

    # ---------------- Add bad channels ----------------#
    filtered_data.info['bads'] += subject.bad_channels

    # ---------------- Save preprocesed data ----------------#
    save.preprocessed_data(raw=filtered_data,
                           et_data_scaled=et_channels_meg,
                           subject=subject,
                           config=config)

    ##
    # ---------------- Fixations and saccades detection ----------------#
    fixations, saccades, pursuit, subject = functions_preproc.fixations_saccades_detection(raw=raw,
                                                                                           et_channels_meg=et_channels_meg,
                                                                                           subject=subject)

    # ---------------- Defining response events and trials from triggers ----------------#
    raw, subject = functions_preproc.define_events_trials_trig(raw=raw,
                                                               subject=subject,
                                                               exp_info=exp_info)

    # ---------------- Saccades classification ----------------#
    saccades, raw, subject = functions_preproc.saccades_classification(subject=subject,
                                                                       saccades=saccades,
                                                                       raw=raw)

    # ---------------- Fixations classification ----------------#
    fixations, raw = functions_preproc.fixation_classification(subject=subject,
                                                               fixations=fixations,
                                                               raw=raw)

    # ---------------- Items classification ----------------#
    raw, subject, ms_items_pos = functions_preproc.ms_items_fixations(fixations=fixations,
                                                                      subject=subject,
                                                                      raw=raw,
                                                                      distance_threshold=80)

    raw, subject, items_pos = functions_preproc.target_vs_distractor(fixations=fixations,
                                                                     subject=subject,
                                                                     raw=raw,
                                                                     distance_threshold=80)

    # ---------------- Save fix time distribution, pupils size vs mss, scanpath and trial gaze figures ----------------#
    if plot:
        plot_preproc.first_fixation_delay(subject=subject)
        plot_preproc.pupil_size_increase(subject=subject)
        plot_preproc.performance(subject=subject)
        plot_preproc.fixation_duration(subject=subject)
        plot_preproc.saccades_amplitude(subject=subject)
        plot_preproc.saccades_dir_hist(subject=subject)
        plot_preproc.sac_main_seq(subject=subject)

    # ---------------- Add scaled ET data to MEG data as new channels ----------------#
    raw = functions_preproc.add_et_channels(raw=raw,
                                            et_channels_meg=et_channels_meg,
                                            et_channel_names=subject.et_channel_names)

    # ---------------- Filter line noise ----------------#
    filtered_data = functions_preproc.filter_line_noise(subject=subject,
                                                        raw=raw,
                                                        freqs=subject.line_noise_freqs)

    # ---------------- Add bad channels ----------------#
    filtered_data.info['bads'] += subject.bad_channels

    # ---------------- Add clean annotations to meg data if already annotated ----------------#
    '''
    This is in case of re-running preprocessing module starting from raw MEG data, and a manual artifact annotation was already saved for that subject.
    This will include those annotations in the new preprocessed data and update the Annot_PSD plot.
    '''
    preproc_data_path = paths.preproc_path
    preproc_save_path = preproc_data_path + subject.subject_id + '/'
    file_path = preproc_save_path + 'clean_annotations.csv'
    if os.path.exists(file_path):
        clean_annotations = mne.read_annotations(fname=file_path)
        filtered_data.set_annotations(clean_annotations)

        # ---------------- Plot new PSD from annotated data ----------------#
        fig = filtered_data.plot_psd(picks='mag')
        fig_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + '/'
        fig_name = 'Annot_PSD'
        save.fig(fig=fig, path=fig_path, fname=fig_name)

    # ---------------- Interpolate bads if any ----------------#
    if len(filtered_data.info['bads']) > 0:
        # Set digitalization info in meg_data
        filtered_data = functions_preproc.set_digitlization(subject=subject,
                                                            meg_data=filtered_data)

        # Interpolate channels
        filtered_data.interpolate_bads()

    # ---------------- Save preprocesed data ----------------#
    save.preprocessed_data(raw=filtered_data,
                           et_data_scaled=et_channels_meg,
                           subject=subject,
                           config=config)

    # ---------------- Free up memory ----------------#
    del (raw)
    del (filtered_data)
    del (subject)
