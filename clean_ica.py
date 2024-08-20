import os
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import mne
import pathlib
import save
import paths
import setup
import load
import functions_analysis

plt.figure()
plt.close('all')

preproc_path = paths.preproc_path
ica_path = paths.ica_path
plot_path = paths.plots_path
exp_info = setup.exp_info()

plot_components = True
for subject_code in exp_info.subjects_ids:

    # Load data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg_data()

    # Downsample and filter
    sfreq = 200
    lfreq = 1
    hfreq = 40
    downsampled_path = pathlib.Path(os.path.join(preproc_path, subject.subject_id, f'down-filt({sfreq}_{lfreq}_{hfreq})_meg.fif'))
    loaded_data = False

    try:
        # Load
        meg_downsampled = mne.io.read_raw_fif(downsampled_path, preload=False)
        loaded_data = True
        print('Downsampled data laoded')
    except:
        # Compute and save
        print('Downsampling and filtering...')
        meg_downsampled = meg_data.copy().pick_types(meg=True)
        meg_downsampled.resample(sfreq)
        meg_downsampled.filter(lfreq, hfreq)
        meg_downsampled.save(downsampled_path, overwrite=True)

    # ICA
    save_path_ica = ica_path + subject.subject_id + '/'
    ica_fname = 'ICA.pkl'
    ica_components = 64

    try:
        if loaded_data:
            # Load ICA
            ica = load.var(file_path=save_path_ica + ica_fname)
            print('ICA object loaded')
        else:
            print('No loaded ICA. Running ICA on new data')
            raise ValueError
    except:
        # Define ICA
        print('Running ICA...')
        ica = ICA(method='infomax', random_state=97, n_components=ica_components)

        # Apply ICA
        ica.fit(meg_downsampled)

        # Save ICA
        os.makedirs(save_path_ica, exist_ok=True)
        save.var(var=ica, path=save_path_ica, fname=ica_fname)

    # Load epochs and evoked data from saccades to plot components and check saccadic artifacts by plot_overlay
    # ID
    epoch_id = 'l_sac'
    # Get time windows from epoch_id name
    tmin, tmax = -0.05, 0.1

    # Epoch data and make evoked
    epochs, events = functions_analysis.epoch_data(subject=subject, meg_data=meg_data, epoch_id=epoch_id, tmin=tmin, tmax=tmax)
    evoked = epochs.average()

    # Ploch's algorithm for saccadic artifacts detection by variance comparison
    ocular_components, sac_epochs_ds, fix_epochs_ds = functions_analysis.ocular_components_ploch(subject=subject, meg_downsampled=meg_downsampled,
                                                                                                 ica=ica, plot_distributions=False)

    # Components figures paths
    fig_path = plot_path + f'ICA/{subject.subject_id}/'
    os.makedirs(fig_path, exist_ok=True)
    # Excluded components save path
    fig_path_ex = fig_path + 'Excluded/'
    os.makedirs(fig_path, exist_ok=True)

    if plot_components:
        # Plot and save components and properties
        plt.ioff()
        ica.plot_components(show=False)

        # Get figures and save
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for i, fig in enumerate(figs):
            save.fig(fig=fig, path=fig_path, fname=f'Components_{i}')
        plt.close('all')

        # Plot properties on all data and save
        # Get sources data to plot time series
        time_series_plot_time = 10
        sources = ica.get_sources(inst=meg_downsampled, start=int(len(meg_downsampled.times)/2), stop=int(len(meg_downsampled.times)/2 + meg_downsampled.info['sfreq'] *time_series_plot_time))

        for ic in range(ica_components):
            functions_analysis.plot_ica_properties(ica=ica, meg_downsampled=meg_downsampled, sources=sources, ic=ic, hfreq=hfreq, epochs=epochs, fig_path=fig_path)

    # Visual inspection of sources for further artefactual components identification
    ica.plot_sources(meg_downsampled, title='ICA', block=True)

    # Select bad components by variable
    ex_components = []

    # Select bad components by input
    if not len(ex_components):
        answer = None
        while answer != 'y':
            answer = input('Enter the component numbers to exclude separated by dashes\n'
                           'For example, to exclude 0th, 1st and 5th components enter: 0-1-5')

            ex_components = answer.split('-')

            try:
                ex_components = [int(comp) for comp in ex_components]
                answer = input(f'The components to exclude are: {ex_components}\n'
                               f'Is that correct? (y/n)')
            except:
                print(f'Error to convert components to integer values.\n'
                      f'components: {ex_components}\n'
                      f'Please re-enter the components to exclude')
                answer = None

    # Append ocular components from Ploch's algorithm to the components to exclude
    for ocular_component in ocular_components:
        if ocular_component not in ex_components:
            ex_components.append(ocular_component)

    # Plot evoked before and after
    before_after = ica.plot_overlay(inst=evoked, exclude=ex_components, show=True)
    save.fig(fig=before_after, path=fig_path_ex, fname=f'Raw_ICA')
    plt.close('all')

    # Plot properties of excluded components
    plt.ioff()

    for ic in ex_components:
        functions_analysis.plot_ica_properties(ica=ica, meg_downsampled=meg_downsampled, sources=sources, ic=ic, hfreq=hfreq, epochs=epochs, fig_path=fig_path_ex)

    # Exclude bad components from data
    ica.exclude = ex_components
    subject.ex_components = ex_components
    meg_ica = meg_data.copy()
    meg_ica.load_data()
    ica.apply(meg_ica)

    # Save ICA clean data
    save_path_ica = ica_path + subject.subject_id + '/'
    os.makedirs(save_path_ica, exist_ok=True)
    path_file = os.path.join(save_path_ica, f'Subject_{subject.subject_id}_ICA.fif')
    meg_ica.save(path_file, overwrite=True)

    # Save subject
    save.var(var=subject, path=save_path_ica, fname='Subject_data.pkl')


## Ploch's algorithm on Max variance

# ocular_components_v, sac_variance_v, fix_variance = \
#     functions_analysis.ocular_components_ploch(subject=subject, meg_downsampled=meg_downsampled,
#                                                sac_id='u_sac_emap', ica=ica)
# ocular_components_h, sac_variance_h, _ = \
#     functions_analysis.ocular_components_ploch(subject=subject, meg_downsampled=meg_downsampled,
#                                                sac_id='r_sac_emap', ica=ica)
#
# # Compute mean component variances
# max_sac_variance_v = np.max(sac_variance_v, axis=0)
# max_sac_variance_h = np.max(sac_variance_h, axis=0)
# max_fix_variance = np.max(fix_variance, axis=0)
#
# max_sac_variances = np.vstack((max_sac_variance_v, max_sac_variance_h))
# max_sac_variance = np.max(max_sac_variances, axis=0)
#
# # Compute variance ratio
# variance_ratio = max_sac_variance / max_fix_variance
#
# # Compute artifactual components
# threshold = 1.1
# ocular_components_max_var = np.where(variance_ratio > threshold)[0]
#
# print('The ocular components to exclude based on the variance ration between saccades and fixations with a '
#       f'threshold of {threshold} are: {ocular_components_max_var}')