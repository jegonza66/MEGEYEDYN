import mne
import numpy as np
import functions_general
import matplotlib.pyplot as plt
import paths
import os
import time
import save
import load
import setup
from mne.decoding import ReceptiveField
from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc, permutation_cluster_1samp_test



def define_events(subject, meg_data, epoch_id, trial_num=None, evt_dur=None, epoch_keys=None, evt_from_df=False, df=None):
    '''

    :param subject:
    :param meg_data:
    :param epoch_id:
    :param evt_dur:
    :param epoch_keys: List of str indicating epoch_ids to epoch data on. Default: None.
    If not provided, will epoch data based on other parameters. If provided will override all other parameters.
    :return:
    '''

    print('Defining events')

    if 'fix' in epoch_id or 'sac' in epoch_id or 'pur' in epoch_id:
        evt_from_df = True

    if evt_from_df:
        if df is not None:
            metadata = df
        elif 'fix' in epoch_id:
            metadata = subject.fixations
            #     if tgt == 1:
            #         metadata = metadata.loc[(metadata['fix_target'] == tgt)]
            #     elif tgt == 0:
            #         metadata = metadata.loc[(metadata['fix_target'] == tgt)]
        elif 'sac' in epoch_id:
            metadata = subject.saccades
            sac_dir = epoch_id.split('_sac')[0]
            if sac_dir != '' and sac_dir != epoch_id:
                metadata = metadata.loc[(metadata['dir'] == sac_dir)]
        elif 'pur' in epoch_id:
            metadata = subject.pursuit

        # Get events with duration
        if evt_dur:
            metadata = metadata.loc[(metadata['duration'] >= evt_dur)]


        metadata.reset_index(drop=True, inplace=True)

        print('Matching MEG times to events onset (this might take a while...)')
        events_samples, event_times = functions_general.find_nearest(meg_data.times, metadata['onset'])

        events = np.zeros((len(events_samples), 3)).astype(int)
        events[:, 0] = events_samples
        events[:, 2] = metadata.index

        events_id = dict(zip(metadata.index.astype(str), metadata.index))  # dict(zip(metadata.id, metadata.index))

    else:
        # Get events from annotations
        all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)

        if epoch_keys is None:
            # Define epoch keys as all events
            epoch_keys = []

            # Iterate over posible multiple epoch ids
            for epoch_sub_id in epoch_id.split('+'):

                # Select epochs
                epoch_keys += [key for key in all_event_id if epoch_sub_id in key]
                if 'sac' not in epoch_sub_id:
                    epoch_keys = [key for key in epoch_keys if 'sac' not in key]
                if 'fix' not in epoch_sub_id:
                    epoch_keys = [key for key in epoch_keys if 'fix' not in key]
                if trial_num != None and any('_t' in epoch_key for epoch_key in epoch_keys):
                    try:
                        if 'vsend' in epoch_sub_id or 'cross1' in epoch_sub_id:
                            epoch_keys = [epoch_key for epoch_key in epoch_keys if epoch_key.split('_t')[-1] in trial_num]
                        else:
                            epoch_keys = [epoch_key for epoch_key in epoch_keys if (epoch_key.split('_t')[-1].split('_')[0] in trial_num and 'end' not in epoch_key)]
                    except:
                        print('Trial selection skipped. Epoch_id does not contain trial number.')

                # Set duration limit
                if 'fix' in epoch_sub_id:
                    metadata = subject.fixations
                    if evt_dur:
                        metadata = metadata.loc[(metadata['duration'] >= evt_dur)]
                        metadata_ids = list(metadata['id'])
                        epoch_keys = [key for key in epoch_keys if key in metadata_ids]
                elif 'sac' in epoch_sub_id:
                    metadata = subject.saccades
                    if evt_dur:
                        metadata = metadata.loc[(metadata['duration'] >= evt_dur)]
                        metadata_ids = list(metadata['id'])
                        epoch_keys = [key for key in epoch_keys if key in metadata_ids]

        # Get events and ids matchig selection
        metadata, events, events_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id, row_events=epoch_keys, tmin=0, tmax=0, sfreq=meg_data.info['sfreq'])

    return metadata, events, events_id

def epoch_data(subject, meg_data, epoch_id, tmin, tmax, trial_num=None, evt_dur=None, baseline=(None, 0), reject=None, evt_from_df=False, df=None,
               save_data=False, epochs_save_path=None, epochs_data_fname=None):
    '''
    :param subject:
    :param mss:
    :param corr_ans:
    :param tgt_pres:
    :param epoch_id:
    :param meg_data:
    :param tmin:
    :param tmax:
    :param baseline: tuple
    Baseline start and end times.
    :param reject: float|str|bool
    Peak to peak amplituyde reject parameter. Use 'subject' for subjects default calculated for short fixation epochs.
     Use False for no rejection. Default to 4e-12 for magnetometers.
    :param save_data:
    :param epochs_save_path:
    :param epochs_data_fname:
    :return:
    '''

    # Sanity check to save data
    if save_data and (not epochs_save_path or not epochs_data_fname):
        raise ValueError('Please provide path and filename to save data. If not, set save_data to false.')

    # Define events
    metadata, events, events_id = define_events(subject=subject, meg_data=meg_data, epoch_id=epoch_id, evt_dur=evt_dur, trial_num=trial_num,
                                                evt_from_df=evt_from_df, df=df)

    # Reject based on channel amplitude
    if reject == False:
        # Setting reject parameter to False uses No rejection (None in mne will not reject)
        reject = None
    elif reject == 'subject':
        reject = dict(mag=subject.config.general.reject_amp)
    elif reject == None:
        reject = dict(mag=5e-12)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                        event_repeated='drop', metadata=metadata, preload=True, baseline=baseline)
    # Drop bad epochs
    epochs.drop_bad()

    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        os.makedirs(epochs_save_path, exist_ok=True)
        epochs.save(epochs_save_path + epochs_data_fname, overwrite=True)

    return epochs, events


def time_frequency(epochs, l_freq, h_freq, freqs_type='lin', n_cycles_div=2., average=True, return_itc=True, output='power',
                   save_data=False, trf_save_path=None, power_data_fname=None, itc_data_fname=None, n_jobs=4):

    # Sanity check to save data
    if save_data and (not trf_save_path or not power_data_fname) or (return_itc and not itc_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')

    # Compute power over frequencies
    print('Computing power and ITC')
    if freqs_type == 'log':
        freqs = np.logspace(np.log10([l_freq, h_freq]), num=40)
    elif freqs_type == 'lin':
        freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)  # 1 Hz bands
    n_cycles = freqs / n_cycles_div  # different number of cycle per frequency
    if return_itc:
        power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                   average=average, output=output,
                                                   return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)
        if save_data:
            # Save trf data
            os.makedirs(trf_save_path, exist_ok=True)
            power.save(trf_save_path + power_data_fname, overwrite=True)
            itc.save(trf_save_path + itc_data_fname, overwrite=True)

        return power, itc

    else:
        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=average,
                                              output=output, return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)
        if save_data:
            # Save trf data
            os.makedirs(trf_save_path, exist_ok=True)
            power.save(trf_save_path + power_data_fname, overwrite=True)

        return power


def time_frequency_multitaper(epochs, l_freq, h_freq, freqs_type='lin', n_cycles_div=2., average=True, return_itc=True,
                               save_data=False, trf_save_path=None, power_data_fname=None, itc_data_fname=None, n_jobs=4):

    # Sanity check to save data
    if save_data and (not trf_save_path or not power_data_fname) or (return_itc and not itc_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')

    # Compute power over frequencies
    print('Computing power and ITC')
    if freqs_type == 'log':
        freqs = np.logspace(np.log10([l_freq, h_freq]), num=40)
    elif freqs_type == 'lin':
        freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)  # 1 Hz bands
    n_cycles = freqs / n_cycles_div  # different number of cycle per frequency
    if return_itc:
        power, itc = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                   average=average,
                                                   return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)
        if save_data:
            # Save trf data
            os.makedirs(trf_save_path, exist_ok=True)
            power.save(trf_save_path + power_data_fname, overwrite=True)
            itc.save(trf_save_path + itc_data_fname, overwrite=True)

        return power, itc

    else:
        power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=average,
                                              return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)
        if save_data:
            # Save trf data
            os.makedirs(trf_save_path, exist_ok=True)
            power.save(trf_save_path + power_data_fname, overwrite=True)

        return power


def get_plot_tf(tfr, plot_xlim=(None, None), plot_max=True, plot_min=True):
    if plot_xlim:
        tfr_crop = tfr.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1])
    else:
        tfr_crop = tfr.copy()

    timefreqs = []

    if plot_max:
        max_ravel = tfr_crop.data.mean(0).argmax()
        freq_idx = int(max_ravel / len(tfr_crop.times))
        time_percent = max_ravel / len(tfr_crop.times) - freq_idx
        time_idx = round(time_percent * len(tfr_crop.times))
        max_timefreq = (tfr_crop.times[time_idx], tfr_crop.freqs[freq_idx])
        timefreqs.append(max_timefreq)

    if plot_min:
        min_ravel = tfr_crop.data.mean(0).argmin()
        freq_idx = int(min_ravel / len(tfr_crop.times))
        time_percent = min_ravel / len(tfr_crop.times) - freq_idx
        time_idx = round(time_percent * len(tfr_crop.times))
        min_timefreq = (tfr_crop.times[time_idx], tfr_crop.freqs[freq_idx])
        timefreqs.append(min_timefreq)

    timefreqs.sort()

    return timefreqs


def ocular_components_ploch(subject, meg_downsampled, ica, sac_id='sac_emap', fix_id='fix_emap' , reject={'mag': 5e-12}, threshold=1.1, plot_distributions=True):
    '''
    Ploch's algorithm for saccadic artifacts detection by variance comparison

    :param subject:
    :param meg_downsampled:
    :param ica:
    :param save_distributions:
    :return: ocular_components
    '''

    # Define events
    print('Saccades')
    sac_metadata, sac_events, sac_events_id = define_events(subject=subject, epoch_id=sac_id, meg_data=meg_downsampled)

    print('Fixations')
    fix_metadata, fix_events, fix_events_id = define_events(subject=subject, epoch_id=fix_id, meg_data=meg_downsampled)

    # Get time windows from epoch_id name
    sac_tmin = -0.005  # Add previous 5 ms
    sac_tmax = sac_metadata['duration'].mean()
    fix_tmin = 0
    fix_tmax = fix_metadata['duration'].mean()

    # Epoch data
    sac_epochs = mne.Epochs(raw=meg_downsampled, events=sac_events, event_id=sac_events_id, tmin=sac_tmin,
                            tmax=sac_tmax, reject=reject,
                            event_repeated='drop', metadata=sac_metadata, preload=True, baseline=(0, 0))
    fix_epochs = mne.Epochs(raw=meg_downsampled, events=fix_events, event_id=fix_events_id, tmin=fix_tmin,
                            tmax=fix_tmax, reject=reject,
                            event_repeated='drop', metadata=fix_metadata, preload=True, baseline=(0, 0))

    # Get the ICA sources for the epoched data
    sac_ica_sources = ica.get_sources(sac_epochs)
    fix_ica_sources = ica.get_sources(fix_epochs)

    # Get the ICA data epoched on the emap saccades
    sac_ica_data = sac_ica_sources.get_data(copy=True)
    fix_ica_data = fix_ica_sources.get_data(copy=True)

    # Compute variance along 3rd axis (time)
    sac_variance = np.var(sac_ica_data, axis=2)
    fix_variance = np.var(fix_ica_data, axis=2)

    # Plot components distributions
    if plot_distributions:
        # Create directory
        plot_path = paths.plots_path
        fig_path = plot_path + f'ICA/{subject.subject_id}/Variance_distributions/'
        os.makedirs(fig_path, exist_ok=True)

        # Disable displaying figures
        plt.ioff()
        time.sleep(1)

        # Plot componets distributions
        print('Plotting saccades and fixations variance distributions')
        for n_comp in range(ica.n_components):
            print(f'\rComponent {n_comp}', end='')
            fig = plt.figure()
            plt.hist(sac_variance[:, n_comp], bins=30, alpha=0.6, density=True, label='Saccades')
            plt.hist(fix_variance[:, n_comp], bins=30, alpha=0.6, density=True, label='Fixations')
            plt.legend()
            plt.title(f'ICA component {n_comp}')

            # Save figure
            save.fig(fig=fig, path=fig_path, fname=f'component_{n_comp}')
            plt.close(fig)
        print()

        # Reenable figures
        plt.ion()

    # Compute mean component variances
    mean_sac_variance = np.mean(sac_variance, axis=0)
    mean_fix_variance = np.mean(fix_variance, axis=0)

    # Compute variance ratio
    variance_ratio = mean_sac_variance / mean_fix_variance

    # Compute artifactual components
    ocular_components = np.where(variance_ratio > threshold)[0]

    print('The ocular components to exclude based on the variance ratio between saccades and fixations with a '
          f'threshold of {threshold} are: {ocular_components}')

    return ocular_components, sac_epochs, fix_epochs


def noise_cov(exp_info, subject, bads, use_ica_data, reject=dict(mag=4e-12), rank=None, high_freq=False):
    '''
    Compute background noise covariance matrix for source estimation.
    :param exp_info:
    :param subject:
    :param meg_data:
    :param use_ica_data:
    :return: noise_cov
    '''

    # Define background noise session id
    noise_date_id = exp_info.subjects_noise[subject.subject_id]

    # Load data
    noise = setup.noise(exp_info=exp_info, date_id=noise_date_id)
    raw_noise = noise.load_preproc_data()

    # Set bads to match participant's
    raw_noise.info['bads'] = bads

    if use_ica_data:
        # ICA
        save_path_ica = paths.ica_path + subject.subject_id + '/'
        ica_fname = 'ICA.pkl'

        # Load ICA
        ica = load.var(file_path=save_path_ica + ica_fname)
        print('ICA object loaded')

        # Get excluded components from subject and apply ICA to background noise
        ica.exclude = subject.ex_components
        # Load raw noise data to apply ICA
        raw_noise.load_data()
        ica.apply(raw_noise)

    # Pick meg channels for source modeling
    raw_noise.pick('meg')

    # Filter in high frequencies
    if high_freq:
        l_freq, h_freq = functions_general.get_freq_band(band_id='HGamma')
        raw_noise = raw_noise.filter(l_freq=l_freq, h_freq=h_freq)

    # Compute covariance to withdraw from meg data
    noise_cov = mne.compute_raw_covariance(raw_noise, reject=reject, rank=rank)

    return noise_cov


def noise_csd(exp_info, subject, bads, use_ica_data, freqs):
    '''
    Compute background noise csd for source estimation.
    :param exp_info:
    :param subject:
    :param meg_data:
    :param use_ica_data:
    :return: noise_cov
    '''

    # Define background noise session id
    noise_date_id = exp_info.subjects_noise[subject.subject_id]

    # Load data
    noise = setup.noise(exp_info=exp_info, date_id=noise_date_id)
    raw_noise = noise.load_preproc_data()

    # Set bads to match participant's
    raw_noise.info['bads'] = bads

    if use_ica_data:
        # ICA
        save_path_ica = paths.ica_path + subject.subject_id + '/'
        ica_fname = 'ICA.pkl'

        # Load ICA
        ica = load.var(file_path=save_path_ica + ica_fname)
        print('ICA object loaded')

        # Get excluded components from subject and apply ICA to background noise
        ica.exclude = subject.ex_components
        # Load raw noise data to apply ICA
        raw_noise.load_data()
        ica.apply(raw_noise)

    # Pick meg channels for source modeling
    raw_noise.pick('mag')

    # Compute covariance to withdraw from meg data
    noise_epoch = mne.Epochs(raw_noise, events=np.array([[0, 0, 0]]), tmin=0, tmax=raw_noise.times[-1], baseline=None, preload=True)
    noise_csd = mne.time_frequency.csd_morlet(epochs=noise_epoch, frequencies=freqs)

    return noise_csd

def get_bad_annot_array(meg_data, subj_path, fname, save_var=True):
    # Get bad annotations times
    bad_annotations_idx = [i for i, annot in enumerate(meg_data.annotations.description) if
                           ('bad' in annot or 'BAD' in annot)]
    bad_annotations_time = meg_data.annotations.onset[bad_annotations_idx]
    bad_annotations_duration = meg_data.annotations.duration[bad_annotations_idx]
    bad_annotations_endtime = bad_annotations_time + bad_annotations_duration

    bad_indexes = []
    for i in range(len(bad_annotations_time)):
        bad_annotation_span_idx = np.where(
            np.logical_and((meg_data.times > bad_annotations_time[i]), (meg_data.times < bad_annotations_endtime[i])))[
            0]
        bad_indexes.append(bad_annotation_span_idx)

    # Flatten all indexes and convert to array
    bad_indexes = functions_general.flatten_list(bad_indexes)
    bad_indexes = np.array(bad_indexes)

    # Make bad annotations binary array
    bad_annotations_array = np.ones(len(meg_data.times))
    bad_annotations_array[bad_indexes] = 0

    # Save arrays
    if save_var:
        save.var(var=bad_annotations_array, path=subj_path, fname=fname)

    return bad_annotations_array

def make_mtrf_input(input_arrays, var_name, subject, meg_data, evt_dur, cond_trials, epoch_keys, bad_annotations_array,
                    subj_path, fname, save_var=True):

    # Define events
    metadata, events, _, = define_events(subject=subject, epoch_id=var_name, evt_dur=evt_dur, meg_data=meg_data,  epoch_keys=epoch_keys)
    # Make input arrays as 0
    input_array = np.zeros(len(meg_data.times))
    # Get events samples index
    evt_idxs = events[:, 0]
    # Set those indexes as 1
    input_array[evt_idxs] = 1
    # Exclude bad annotations
    input_array = input_array * bad_annotations_array
    # Save to all input arrays dictionary
    input_arrays[var_name] = input_array

    # Save arrays
    if save_var:
        save.var(var=input_array, path=subj_path, fname=fname)

    return input_arrays


def fit_mtrf(meg_data, tmin, tmax, alpha, model_input, chs_id, standarize=True, n_jobs=4):

    # Define mTRF model
    rf = ReceptiveField(tmin, tmax, meg_data.info['sfreq'], estimator=alpha, scoring='corrcoef', verbose=False, n_jobs=n_jobs)

    # Get subset channels data as array
    picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
    meg_sub = meg_data.copy().pick(picks)
    meg_data_array = meg_sub.get_data()
    if standarize:
        # Standarize data
        print('Computing z-score...')
        meg_data_array = np.expand_dims(meg_data_array, axis=0)  # Need shape (n_epochs, n_channels n_times)
        meg_data_array = mne.decoding.Scaler(info=meg_sub.info, scalings='mean').fit_transform(meg_data_array)
        meg_data_array = meg_data_array.squeeze()
    # Transpose to input the model
    meg_data_array = meg_data_array.T

    # Fit TRF
    rf.fit(model_input, meg_data_array)

    return rf


def run_source_permutations_test(src, stc, source_data, subject, exp_info, save_regions, fig_path, surf_vol, p_threshold=0.05, n_permutations=1024, desired_tval='TFCE',
                                 mask_negatives=False):

    # Return variables
    stc_all_cluster_vis, significant_voxels, significance_mask, t_thresh_name, time_label = None, None, None, None, None

    # Compute source space adjacency matrix
    print("Computing adjacency matrix")
    adjacency_matrix = mne.spatial_src_adjacency(src)

    # Transpose source_fs_data from shape (subjects x space x time) to shape (subjects x time x space)
    source_data_default = source_data.swapaxes(1, 2)

    # Define the t-value threshold for cluster formation
    if desired_tval == 'TFCE':
        t_thresh = dict(start=0, step=0.1)
    else:
        df = len(exp_info.subjects_ids) - 1  # degrees of freedom for the test
        t_thresh = stats.distributions.t.ppf(1 - desired_tval / 2, df=df)

    # Run permutations
    T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(X=source_data_default,
                                                                                     n_permutations=n_permutations,
                                                                                     adjacency=adjacency_matrix,
                                                                                     n_jobs=4, threshold=t_thresh)

    # Select the clusters that are statistically significant at p
    good_clusters_idx = np.where(cluster_p_values < p_threshold)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    significant_pvalues = [cluster_p_values[idx] for idx in good_clusters_idx]

    if len(good_clusters):

        # variable for figure fnames and p_values as title
        if type(t_thresh) == dict:
            time_label = f'{np.round(np.mean(significant_pvalues), 4)} +- {np.round(np.std(significant_pvalues), 4)}'
            t_thresh_name = 'TFCE'
        else:
            time_label = str(significant_pvalues)
            t_thresh_name = round(t_thresh, 2)

        # Get vertices from source space
        fsave_vertices = [s["vertno"] for s in src]

        # Select clusters for visualization
        stc_all_cluster_vis = summarize_clusters_stc(clu=clu, p_thresh=p_threshold, tstep=stc.tstep, vertices=fsave_vertices, subject=subject)

        # Get significant clusters
        significance_mask = np.where(stc_all_cluster_vis.data[:, 0] == 0)[0]
        significant_voxels = np.where(stc_all_cluster_vis.data[:, 0] != 0)[0]

        # Get significant AAL and brodmann regions from mni space
        if save_regions:
            os.makedirs(fig_path, exist_ok=True)
            significant_regions_df = functions_general.get_regions_from_mni(src_default=src, significant_voxels=significant_voxels, save_path=fig_path, surf_vol=surf_vol,
                                                                            t_thresh_name=t_thresh_name, p_threshold=p_threshold, masked_negatves=mask_negatives)

    return stc_all_cluster_vis, significant_voxels, significance_mask, t_thresh_name, time_label, p_threshold


def estimate_sources_cov(subject, baseline, band_id, filter_sensors, filter_method, use_ica_data, epoch_id, mss, corr_ans, tgt_pres, trial_dur, reject, tmin, tmax,
                         filters, active_times, rank, bline_mode_subj, save_data, cov_save_path, cov_act_fname, cov_baseline_fname, epochs_save_path, epochs_data_fname):

    try:
        # Load covariance matrix
        baseline_cov = mne.read_cov(fname=cov_save_path + cov_baseline_fname)
    except:
        # Load epochs
        try:
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        except:
            # Compute epochs
            if use_ica_data:
                if band_id and filter_sensors:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=save_data,
                                                  method=filter_method)
                else:
                    meg_data = load.ica_data(subject=subject)
            else:
                if band_id and filter_sensors:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, use_ica_data=False,
                                                  save_data=save_data, method=filter_method)
                else:
                    meg_data = subject.load_preproc_meg_data()

            # Epoch data
            epochs, events = epoch_data(subject=subject, mss=mss, corr_ans=corr_ans, tgt_pres=tgt_pres,
                                                           epoch_id=epoch_id, meg_data=meg_data, tmin=tmin, trial_dur=trial_dur,
                                                           tmax=tmax, reject=reject, baseline=baseline,
                                                           save_data=save_data, epochs_save_path=epochs_save_path,
                                                           epochs_data_fname=epochs_data_fname)

        # Compute covariance matrices
        baseline_cov = mne.cov.compute_covariance(epochs=epochs, tmin=baseline[0], tmax=baseline[1], method="shrunk", rank=dict(mag=rank))
        # Save
        if save_data:
            os.makedirs(cov_save_path, exist_ok=True)
            baseline_cov.save(fname=cov_save_path + cov_baseline_fname, overwrite=True)

    try:
        # Load covariance matrix
        active_cov = mne.read_cov(fname=cov_save_path + cov_act_fname)
    except:
        # Load epochs
        try:
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        except:
            # Compute epochs
            if use_ica_data:
                if band_id and filter_sensors:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=save_data,
                                                  method=filter_method)
                else:
                    meg_data = load.ica_data(subject=subject)
            else:
                if band_id and filter_sensors:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, use_ica_data=False,
                                                  save_data=save_data, method=filter_method)
                else:
                    meg_data = subject.load_preproc_meg_data()

            # Epoch data
            epochs, events = epoch_data(subject=subject, mss=mss, corr_ans=corr_ans, tgt_pres=tgt_pres,
                                                           epoch_id=epoch_id, meg_data=meg_data, tmin=tmin, trial_dur=trial_dur,
                                                           tmax=tmax, reject=reject, baseline=baseline,
                                                           save_data=save_data, epochs_save_path=epochs_save_path,
                                                           epochs_data_fname=epochs_data_fname)

        # Compute covariance matrices
        active_cov = mne.cov.compute_covariance(epochs=epochs, tmin=active_times[0], tmax=active_times[1], method="shrunk", rank=dict(mag=rank))
        # Save
        if save_data:
            os.makedirs(cov_save_path, exist_ok=True)
            active_cov.save(fname=cov_save_path + cov_act_fname, overwrite=True)

    # Compute sources and apply baseline
    stc_base = mne.beamformer.apply_lcmv_cov(baseline_cov, filters)
    stc_act = mne.beamformer.apply_lcmv_cov(active_cov, filters)

    if bline_mode_subj == 'mean':
        stc = stc_act - stc_base
    elif bline_mode_subj == 'ratio':
        stc = stc_act / stc_base
    elif bline_mode_subj == 'db':
        stc = stc_act / stc_base
        stc.data = 10 * np.log10(stc.data)
    else:
        stc = stc_act

    return stc



def run_time_frequency_test(data, pval_threshold, t_thresh, min_sig_chs=0, n_permutations=1024):

    # Clusters out type
    if type(t_thresh) == dict:
        out_type = 'indices'
    else:
        out_type = 'mask'

    significant_pvalues = None

    # Permutations cluster test (TFCE if t_thresh as dict)
    t_tfce, clusters, p_tfce, H0 = permutation_cluster_1samp_test(X=data, threshold=t_thresh, n_permutations=n_permutations,
                                                                  out_type=out_type, n_jobs=4)

    # Make clusters mask
    if type(t_thresh) == dict:
        # If TFCE use p-vaues of voxels directly
        p_tfce = p_tfce.reshape(data.shape[-2:])  # Reshape to data's shape
        clusters_mask_plot = p_tfce < pval_threshold
        clusters_mask = None

    else:
        # Get significant clusters
        good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
        significant_clusters = [clusters[idx] for idx in good_clusters_idx]
        significant_pvalues = [p_tfce[idx] for idx in good_clusters_idx]

        # Reshape to data's shape by adding all clusters into one bool array
        clusters_mask = np.zeros(data[0].shape)
        if len(significant_clusters):
            for significant_cluster in significant_clusters:
                clusters_mask += significant_cluster

            if min_sig_chs:
                clusters_mask_plot = clusters_mask.sum(axis=-1) > min_sig_chs
            else:
                clusters_mask_plot = clusters_mask.sum(axis=-1)
            clusters_mask_plot = clusters_mask_plot.astype(bool)

        else:
            clusters_mask_plot = None

    return clusters_mask, clusters_mask_plot, significant_pvalues