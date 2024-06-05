import functions_general
import plot_general
import functions_analysis
import setup
import save
import functions_preproc
import paths
import mne
import matplotlib.pyplot as plt


# Load experiment info
exp_info = setup.exp_info()
# Load configuration
# config = load.config(path=paths.config_path, fname='config.pkl')
config = setup.config()
# Run plots
plot = False

# Run
evokeds = []
for subject_code in exp_info.subjects_ids:

    # ---------------- Load data ----------------#
    # Define subject
    subject = setup.raw_subject(exp_info=exp_info,
                                config=config,
                                subject_code=subject_code)

    # Load Meg data
    raw = subject.load_raw_meg_data()

    # Pick channels
    chs_id = 'mag'

    # Plot PSD
    raw.pick(chs_id)
    fig = raw.plot_psd()
    save.fig(fig=fig, path=paths.plots_path + f'Preprocessing/{subject.subject_id}', fname='PSD')

    # Get ET channels from MEG
    print('\nGetting ET channels data from MEG')
    et_channels_meg = raw.get_data(picks=exp_info.et_channel_names[subject.tracked_eye])

    #---------------- Remove DAC delay samples ----------------#
    meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw = functions_preproc.DAC_samples(et_channels_meg=et_channels_meg,
                                                                                                exp_info=exp_info,
                                                                                                sfreq=raw.info['sfreq'])

    #---------------- Reescaling based on conversion parameters ----------------#
    meg_gazex_data_scaled, meg_gazey_data_scaled = functions_preproc.reescale_et_channels(meg_gazex_data_raw=meg_gazex_data_raw,
                                                                                          meg_gazey_data_raw=meg_gazey_data_raw)

    #---------------- Blinks removal ----------------#
    # Define intervals around blinks to also fill with nan. Due to conversion noise from square signal
    et_channels_meg = functions_preproc.blinks_to_nan(exp_info=exp_info,
                                                      subject=subject,
                                                      meg_gazex_data_scaled=meg_gazex_data_scaled,
                                                      meg_gazey_data_scaled=meg_gazey_data_scaled,
                                                      meg_pupils_data_raw=meg_pupils_data_raw,
                                                      config=subject.config.preproc)

    #---------------- Fixations and saccades detection ----------------#
    fixations, saccades, pursuit, subject = functions_preproc.fixations_saccades_detection(raw=raw,
                                                                                           et_channels_meg=et_channels_meg,
                                                                                           subject=subject)
    # ---------------- Quick analysis ----------------#

    # Define dataframes to compute events from
    subject.fixations = fixations
    subject.saccades = saccades
    subject.pursuit = pursuit

    # Select events
    epoch_id = 'fix'
    tmin = -0.2
    tmax = 0.5

    # Epoch data
    epochs_save_path = paths.save_path + f'Epochs_Raw/{epoch_id}/'
    epochs_data_fname = f'{subject.subject_id}_epo.fif'

    try:
        # Load epoched data
        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
    except:
        # Compute
        epochs, events = functions_analysis.epoch_data(subject, meg_data=raw, epoch_id=epoch_id, tmin=tmin, tmax=tmax, evt_from_df=True,
                                                       save_data=False, epochs_save_path=epochs_save_path, epochs_data_fname=epochs_data_fname)

    # Pick MEG channels to plot
    picks = functions_general.pick_chs(chs_id=chs_id, info=epochs.info)

    # # Parameters for plotting
    # overlay = epochs.metadata['duration']
    # if overlay is not None:
    #     order = overlay.argsort()  # Sorting from longer to shorter
    # else:
    #     order = None
    # sigma = 5
    # combine = 'gfp'
    # group_by = {}
    #
    # # Figure path and name
    # epoch_fig_path = paths.plots_path + f'Epochs_Raw/{epoch_id}/'
    # fname = subject.subject_id + f'_{chs_id}_{combine}'
    #
    # # Plot epochs
    # plot_general.epochs(subject=subject, epochs=epochs, picks=picks, order=order, overlay=overlay, combine=combine, sigma=sigma,
    #                     group_by=group_by, display_figs=False, save_fig=True, fig_path=epoch_fig_path, fname=fname)

    # ----- Evoked -----#
    # Define evoked
    evoked = epochs.average(picks=['mag', 'misc'])
    evokeds.append(evoked)

    # Separete MEG and misc channels
    evoked_meg = evoked.copy().pick('mag')
    evoked_misc = evoked.copy().pick('misc')
    evoked_misc.pick(exp_info.et_channel_names[subject.tracked_eye])

    # Plot evoked
    evoked_fig_path = paths.plots_path + f'Evoked_Raw/{epoch_id}/'
    fname = subject.subject_id + f'_{chs_id}'
    plot_general.evoked(evoked_meg=evoked_meg, evoked_misc=evoked_misc, picks=picks,
                        plot_gaze=True, plot_xlim=(tmin, tmax), display_figs=False, save_fig=True,
                        fig_path=evoked_fig_path, fname=fname)

    plt.close('all')

# Compute grand average
grand_avg = mne.grand_average(evokeds)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')
grand_avg_misc = grand_avg.copy().pick('misc')
grand_avg_misc.pick(exp_info.et_channel_names[subject.tracked_eye])

# Plot evoked
fname = f'GA_{chs_id}'
# ylim = dict({'mag': (-150, 200)})
plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=chs_id,
                    plot_gaze=True, plot_xlim=(tmin, tmax), display_figs=False, save_fig=True,
                    fig_path=evoked_fig_path, fname=fname)
