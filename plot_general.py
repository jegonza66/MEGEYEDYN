import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import paths
import save
import functions_general
import functions_analysis
import mne
import mne_connectivity
from nilearn import plotting
from itertools import compress


save_path = paths.save_path
plot_path = paths.plots_path


def epochs(subject, epochs, picks, order=None, overlay=None, combine='mean', sigma=5, group_by=None, cmap='bwr',
           vmin=None, vmax=None, display_figs=True, save_fig=None, fig_path=None, fname=None):

    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    fig_ep = epochs.plot_image(picks=picks, order=order, sigma=sigma, cmap=cmap, overlay_times=overlay, combine=combine,
                               vmin=vmin, vmax=vmax, title=subject.subject_id, show=display_figs)

    # Save figure
    if save_fig:
        if len(fig_ep) == 1:
            fig = fig_ep[0]
            save.fig(fig=fig, path=fig_path, fname=fname)
        else:
            for i in range(len(fig_ep)):
                fig = fig_ep[i]
                group = group_by.keys()[i]
                fname += f'{group}'
                save.fig(fig=fig, path=fig_path, fname=fname)


def evoked(evoked_meg, evoked_misc, picks, plot_gaze=False, fig=None,
           axes=None, plot_xlim='tight', plot_ylim=None, display_figs=False, save_fig=True, fig_path=None, fname=None):
    '''
    Plot evoked response with mne.Evoked.plot() method. Option to plot gaze data on subplot.

    :param evoked_meg: Evoked with picked mag channels
    :param evoked_misc: Evoked with picked misc channels (if plot_gaze = True)
    :param picks: Meg channels to plot
    :param plot_gaze: Bool
    :param fig: Optional. Figure instance
    :param axes: Optional. Axes instance (if figure provided)
    :param plot_xlim: tuple. x limits for evoked and gaze plot
    :param plot_ylim: dict. Possible keys: meg, mag, eeg, misc...
    :param display_figs: bool. Whether to show figures or not
    :param save_fig: bool. Whether to save figures. Must provide save_path and figure name.
    :param fig_path: string. Optional. Path to save figure if save_fig true
    :param fname: string. Optional. Filename if save_fig is True

    :return: None
    '''

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if axes:
        if save_fig and not fig:
            raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

        evoked_meg.plot(picks=picks, gfp=True, axes=axes, time_unit='s', spatial_colors=True, xlim=plot_xlim,
                        ylim=plot_ylim, show=display_figs)
        axes.vlines(x=0, ymin=axes.get_ylim()[0], ymax=axes.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    elif plot_gaze:
        # Get Gaze x ch
        gaze_x_ch_idx = np.where(np.array(evoked_misc.ch_names) == 'UADC001-4123')[0][0]
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        axs[1].plot(evoked_misc.times, evoked_misc.data[gaze_x_ch_idx, :])
        axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
        axs[1].set_ylabel('Gaze x')
        axs[1].set_xlabel('Time')

        evoked_meg.plot(picks=picks, gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim,
                        ylim=plot_ylim, show=display_figs)
        axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    else:
        fig = evoked_meg.plot(picks=picks, gfp=True, axes=axes, time_unit='s', spatial_colors=True, xlim=plot_xlim,
                              ylim=plot_ylim, show=display_figs)
        axes = fig.get_axes()[0]
        axes.vlines(x=0, ymin=axes.get_ylim()[0], ymax=axes.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)


def evoked_topo(evoked_meg, picks, topo_times, title=None, fig=None, axes_ev=None, axes_topo=None, xlim=None, ylim=None,
                display_figs=False, save_fig=False, fig_path=None, fname=None):

    # Sanity check
    if save_fig and (not fig_path or not fname):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if axes_ev and axes_topo:
        if save_fig and not fig:
            raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

        display_figs = True

        evoked_meg.plot_joint(times=topo_times, title=title, picks=picks, show=display_figs,
                              ts_args={'axes': axes_ev, 'xlim': xlim, 'ylim': ylim},
                              topomap_args={'axes': axes_topo})

        axes_ev.vlines(x=0, ymin=axes_ev.get_ylim()[0], ymax=axes_ev.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    else:
        fig = evoked_meg.plot_joint(times=topo_times, title=title, picks=picks, show=display_figs,
                                    ts_args={'xlim': xlim, 'ylim': ylim})

        all_axes = plt.gcf().get_axes()
        axes_ev = all_axes[0]
        axes_ev.vlines(x=0, ymin=axes_ev.get_ylim()[0], ymax=axes_ev.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)


def plot_ica_properties(ica, meg_downsampled, sources, ic, hfreq, epochs, fig_path):

    # Create figure
    fig = plt.figure(figsize=(8.0, 9.0), facecolor=[0.95] * 3)

    # Set axes to plot ICA properties
    axes_params = (
        ("topomap", [0.08, 0.6, 0.3, 0.35]),
        ("image", [0.5, 0.65, 0.45, 0.30]),
        ("erp", [0.5, 0.6, 0.45, 0.05]),
        ("spectrum", [0.08, 0.29, 0.32, 0.25]),
        ("variance", [0.5, 0.29, 0.45, 0.25]),
        ("time_series", [0.08, 0.06, 0.87, 0.15])
    )

    # Get axes lists
    axes = [fig.add_axes(loc, label=name) for name, loc in axes_params]
    mne_axes = axes[:-1]
    ax_time_series = axes[-1]

    # Plot properties
    ica.plot_properties(meg_downsampled, picks=ic, psd_args=dict(fmax=hfreq), show=False, axes=mne_axes)

    # Plot time series
    ax_time_series.plot(sources[ic][1], sources[ic][0].ravel(), color='black', linewidth=0.5)
    ax_time_series.set_title('Time series')
    ax_time_series.set_xlabel('Time (s)')

    # Plot epochs on axes
    # Make fake figure to pass for plotting spureous plots
    fake_fig, fake_axs = plt.subplots(nrows=3)

    # Get figure epochs and ERP axes
    image_ax, erp_ax = fig.get_axes()[1], fig.get_axes()[2]
    image_ax.clear()
    erp_ax.clear()

    # Define axes list to pass and plot properties
    ax_list = [fake_axs[0], image_ax, erp_ax, fake_axs[1], fake_axs[2]]

    # Plot epoch properties
    ica.plot_properties(epochs, picks=[ic], axes=ax_list, psd_args=dict(fmax=hfreq), show=False)
    save.fig(fig=fig, path=fig_path, fname=f'IC_{ic}_Properties')
    plt.close('all')



def fig_psd():

    fig = plt.figure(figsize=(15, 5))

    # 1st row Topoplots
    ax1 = fig.add_axes([0.05, 0.6, 0.15, 0.3])
    ax2 = fig.add_axes([0.225, 0.6, 0.15, 0.3])
    ax3 = fig.add_axes([0.4, 0.6, 0.15, 0.3])
    ax4 = fig.add_axes([0.575, 0.6, 0.15, 0.3])
    ax5 = fig.add_axes([0.75,  0.6, 0.15, 0.3])

    # 2nd row PSD
    ax6 = fig.add_axes([0.15,  0.1, 0.7, 0.4])

    # Group axes
    axs_topo = [ax1, ax2, ax3, ax4, ax5]
    ax_psd = ax6

    return fig, axs_topo, ax_psd


def fig_tf_bands(fontsize=None, ticksize=None):

    if fontsize:
        matplotlib.rc({'font.size': fontsize})
    if ticksize:
        matplotlib.rc({'xtick.labelsize': ticksize})
        matplotlib.rc({'ytick.labelsize': ticksize})

    fig, axes_topo = plt.subplots(3, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [5, 1, 1]})

    for ax in axes_topo[:, 0]:
        ax.remove()
    axes_topo = [ax for ax_arr in axes_topo[:, 1:] for ax in ax_arr ]

    ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.8])

    return fig, axes_topo, ax1


def tfr_bands(tfr, chs_id, plot_xlim=(None, None), baseline=None, bline_mode=None, dB=False, vmin=None, vmax=None, subject=None, title=None, vlines_times=[0],
              topo_times=None, display_figs=False, save_fig=False, fig_path=None, fname=None, fontsize=None, ticksize=None, cmap='bwr'):

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if plot_xlim == (None, None):
        plot_xlim = (tfr.tmin, tfr.tmax)

    # Turn off dB if baseline mode is incompatible with taking log10
    if dB and bline_mode in ['mean', 'logratio']:
        dB = False

    # Define figure
    fig, axes_topo, ax_tf = fig_tf_bands(fontsize=fontsize, ticksize=ticksize)

    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=tfr.info)

    # Plot time-frequency
    tfr.plot(picks=picks, baseline=baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
             combine='mean', cmap=cmap, axes=ax_tf, show=display_figs, vmin=vmin, vmax=vmax, dB=dB)

    # Plot time markers as vertical lines
    for t in vlines_times:
        try:
            ax_tf.vlines(x=t, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='gray')
        except:
            pass

    # Topomaps parameters
    if not topo_times:
        topo_times = plot_xlim
    topomap_kw = dict(ch_type='mag', tmin=topo_times[0], tmax=topo_times[1], baseline=baseline,
                      mode=bline_mode, show=display_figs)
    plot_dict = dict(Delta=dict(fmin=1, fmax=4), Theta=dict(fmin=4, fmax=8), Alpha=dict(fmin=8, fmax=12),
                     Beta=dict(fmin=12, fmax=30), Gamma=dict(fmin=30, fmax=45), HGamma=dict(fmin=45, fmax=100))

    # Plot topomaps
    for ax, (title_topo, fmin_fmax) in zip(axes_topo, plot_dict.items()):
        try:
            tfr.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
        except:
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title_topo)

    # Figure title
    if title:
        fig.suptitle(title)
    elif subject:
        fig.suptitle(subject.subject_id + f'_{fname.split("_")[0]}_{chs_id}_{bline_mode}_topotimes_{topo_times}')
    elif not subject:
        fig.suptitle(f'Grand_average_{fname.split("_")[0]}_{chs_id}_{bline_mode}_topotimes_{topo_times}')
        fname = 'GA_' + fname

    fig.tight_layout()

    if save_fig:
        fname += f'_topotimes_{topo_times}'
        os.makedirs(fig_path, exist_ok=True)
        save.fig(fig=fig, path=fig_path, fname=fname)


def fig_tf_times(time_len, figsize=(18, 5), ax_len_div=12, timefreqs_tfr=None, fontsize=None, ticksize=None):

    if fontsize:
        matplotlib.rc({'font.size': fontsize})
    if ticksize:
        matplotlib.rc({'xtick.labelsize': ticksize})
        matplotlib.rc({'ytick.labelsize': ticksize})


    fig = plt.figure(figsize=figsize)

    # # Define axes for topomaps
    if timefreqs_tfr:
        axes_topo = []
        for i in range(len(timefreqs_tfr)):
            width = 0.1 + 0.05
            start_pos = 0.075 + i * width
            ax = fig.add_axes([start_pos, 0.05, 0.1, 0.3])
            axes_topo.append(ax)

        # Define axis for colorbar
        start_pos = 0.075 + (i + 1) * width
        ax_cbar = fig.add_axes([start_pos, 0.05, 0.005, 0.25])
    else:
        axes_topo = None
        ax_cbar = None

    # Define TFR axis
    ax_len = time_len/ax_len_div
    ax_tfr = fig.add_axes([0.075, 0.55, ax_len, 0.3])

    return fig, axes_topo, ax_tfr, ax_cbar


def tfr_times(tfr, chs_id, timefreqs_tfr=None, plot_xlim=(None, None), baseline=None, bline_mode=None, dB=False, vmin=None, vmax=None,
              topo_vmin=None, topo_vmax=None, subject=None, title=None, vlines_times=None, cmap='bwr',
              display_figs=False, save_fig=False, fig_path=None, fname=None, fontsize=None, ticksize=None):

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if plot_xlim == (None, None):
        plot_xlim = (tfr.tmin, tfr.tmax)

    # Turn off dB if baseline mode is incompatible with taking log10
    if dB and bline_mode in ['mean', 'logratio']:
        dB = False

    # Define figure
    time_len = plot_xlim[1] - plot_xlim[0]
    fig, axes_topo, ax_tf, ax_cbar = fig_tf_times(time_len=time_len, timefreqs_tfr=timefreqs_tfr, fontsize=fontsize, ticksize=ticksize)

    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=tfr.info)

    # Plot time-frequency
    tfr_plot = tfr.copy().apply_baseline(baseline=baseline, mode=bline_mode)
    tfr_plot.plot(picks=picks, tmin=plot_xlim[0], tmax=plot_xlim[1], combine='mean', cmap=cmap, axes=ax_tf,
             show=display_figs, vmin=vmin, vmax=vmax, dB=dB)

    # Plot time markers as vertical lines
    if not vlines_times:
        vlines_times = [0]
    ymin = ax_tf.get_ylim()[0]
    ymax = ax_tf.get_ylim()[1]
    for t in vlines_times:
        try:
            ax_tf.vlines(x=t, ymin=ymin, ymax=ymax, linestyles='--', colors='gray')
        except:
            pass

    # Plot topomaps
    if axes_topo:
        # Get min and max from all topoplots
        if topo_vmin == None and topo_vmax == None:
            maxs = []
            if type(timefreqs_tfr) == dict:
                for topo_timefreqs in timefreqs_tfr.items():
                    tfr_crop = tfr_plot.copy().crop(tmin=topo_timefreqs[1]['tmin'], tmax=topo_timefreqs[1]['tmax'],
                                                    fmin=topo_timefreqs[1]['fmin'], fmax=topo_timefreqs[1]['fmax'])
                    data = tfr_crop.data.ravel()
                    maxs.append(data.max())
            elif type(timefreqs_tfr) == list:
                for topo_timefreqs in timefreqs_tfr:
                    tfr_crop = tfr_plot.copy().crop(tmin=topo_timefreqs[0], tmax=topo_timefreqs[0],
                                                    fmin=topo_timefreqs[1], fmax=topo_timefreqs[1])
                    data = tfr_crop.data.ravel()
                    maxs.append(data.max())
            topo_vmax = np.max(maxs)
            topo_vmin = -topo_vmax

        if type(timefreqs_tfr) == dict:
            for i, (ax, (key, topo_timefreqs)) in enumerate(zip(axes_topo, timefreqs_tfr.items())):
                # Topomaps parameters
                topomap_kw = dict(ch_type='mag', tmin=topo_timefreqs['tmin'], tmax=topo_timefreqs['tmax'],
                                  fmin=topo_timefreqs['fmin'], fmax=topo_timefreqs['fmax'], vlim=(topo_vmin, topo_vmax),
                                  cmap=cmap, colorbar=False, baseline=baseline,  mode=bline_mode, show=display_figs)

                try:
                    tfr.plot_topomap(axes=ax, **topomap_kw)
                except:
                    ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
                    ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(topo_timefreqs['title'], color=f'C{i}', fontweight='bold')

        elif type(timefreqs_tfr) == list:
            for i, (ax, (topo_timefreqs)) in enumerate(zip(axes_topo, timefreqs_tfr)):
                # Topomaps parameters
                topomap_kw = dict(ch_type='mag', tmin=topo_timefreqs[0], tmax=topo_timefreqs[0],
                                  fmin=topo_timefreqs[1], fmax=topo_timefreqs[1], vlim=(topo_vmin, topo_vmax),
                                  cmap=cmap, colorbar=False, baseline=baseline, mode=bline_mode, show=display_figs)

                try:
                    tfr.plot_topomap(axes=ax, **topomap_kw)
                except:
                    ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
                    ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(topo_timefreqs, color=f'C{i}', fontweight='bold')

        # Colorbar
        norm = matplotlib.colors.Normalize(vmin=topo_vmin, vmax=topo_vmax)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        # Get colorbar axis
        fig.colorbar(sm, cax=ax_cbar)

    # Figure title
    if timefreqs_tfr:
        try:
            topo_times = [timefreq['tmin'] for (key, timefreq) in timefreqs_tfr.items()]
        except:
            topo_times = [timefreq[0] for timefreq in timefreqs_tfr]
    else:
        topo_times = None

    if title is None:
        title = fname + f'_topotimes_{topo_times}'
    fig.suptitle(title)

    if save_fig:
        fname += f'_topotimes_{topo_times}'
        os.makedirs(fig_path, exist_ok=True)
        save.fig(fig=fig, path=fig_path, fname=fname)


def tfr_plotjoint(tfr, plot_baseline=None, bline_mode=None, plot_xlim=(None, None), timefreqs=None, plot_max=True, plot_min=True, vlines_times=None, cmap='bwr',
                  vmin=None, vmax=None, display_figs=False, save_fig=False, trf_fig_path=None, fname=None, fontsize=None, ticksize=None):
    # Sanity check
    if save_fig and (not fname or not trf_fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if fontsize:
        matplotlib.rc({'font.size': fontsize})
    if ticksize:
        matplotlib.rc({'xtick.labelsize': ticksize})
        matplotlib.rc({'ytick.labelsize': ticksize})

    if plot_baseline:
        tfr_plotjoint = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)
    else:
        tfr_plotjoint = tfr.copy()

    # Get all mag channels to plot
    picks = functions_general.pick_chs(chs_id='mag', info=tfr.info)
    tfr_plotjoint = tfr_plotjoint.pick(picks)

    # Get maximum
    if not timefreqs:
        timefreqs = functions_analysis.get_plot_tf(tfr=tfr_plotjoint, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min)

    # Title
    if fname:
        title = f'{fname.split("_")[1]}_{bline_mode}'
    else:
        title = f'{bline_mode}'

    fig = tfr_plotjoint.plot_joint(timefreqs=timefreqs, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap=cmap, vmin=vmin, vmax=vmax,
                                   title=title, show=display_figs)

    # Plot vertical lines
    tf_ax = fig.axes[0]
    if not vlines_times:
        vlines_times = [0]
    for t in vlines_times:
        try:
            tf_ax.vlines(x=t, ymin=tf_ax.get_ylim()[0], ymax=tf_ax.get_ylim()[1], linestyles='--', colors='gray')
        except:
            pass

    if save_fig:
        save.fig(fig=fig, path=trf_fig_path, fname=fname)


def tfr_plotjoint_picks(tfr, plot_baseline=None, bline_mode=None, plot_xlim=(None, None), timefreqs=None, image_args=None, clusters_mask=None,
                        plot_max=True, plot_min=True, vmin=None, vmax=None, chs_id='mag', vlines_times=None, cmap='bwr',
                        display_figs=False, save_fig=False, trf_fig_path=None, fname=None, fontsize=18, ticksize=18):
    # Sanity check
    if save_fig and (not fname or not trf_fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if fontsize:
        plt.rcParams.update({'font.size': fontsize})
    if ticksize:
        plt.rcParams.update({'xtick.labelsize': ticksize})
        plt.rcParams.update({'ytick.labelsize': ticksize})

    if plot_baseline:
        tfr_plotjoint = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)
        tfr_topo = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)  # tfr for topoplots
    else:
        tfr_plotjoint = tfr.copy()
        tfr_topo = tfr.copy()  # tfr for topoplots

    # TFR from certain chs and topoplots from all channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=tfr.info)
    tfr_plotjoint = tfr_plotjoint.pick(picks)

    # Get maximum
    if timefreqs == None:
        timefreqs = functions_analysis.get_plot_tf(tfr=tfr_plotjoint, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min)

    # Title
    if fname:
        title = f'{fname.split("_")[1]}_{bline_mode}'
    else:
        title = f'{bline_mode}'

    # Get min and max from all topoplots and use in TF plot aswell
    if vmin == None or vmax == None:
        maxs = []
        mins = []
        for timefreq in timefreqs:
            tfr_crop = tfr_topo.copy().crop(tmin=timefreq[0], tmax=timefreq[0], fmin=timefreq[1], fmax=timefreq[1])
            data = tfr_crop.data.ravel()
            mins.append(- 2 * data.std())
            maxs.append(2 * data.std())
        vmax = np.max(maxs)
        vmin = np.min(mins)

    # Plot tf plot joint
    fig = tfr_plotjoint.plot_joint(timefreqs=timefreqs, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap=cmap, image_args=image_args,
                                   title=title, show=display_figs, vmin=vmin, vmax=vmax)

    # Plot vertical lines
    tf_ax = fig.axes[-2]
    if not vlines_times:
        vlines_times = [0]
    for t in vlines_times:
        try:
            tf_ax.vlines(x=t, ymin=tf_ax.get_ylim()[0], ymax=tf_ax.get_ylim()[1], linestyles='--', colors='gray')
        except:
            pass

    # Define Topo mask
    if clusters_mask is not None:
        # Define significant channels to mask in time freq interval around desired tf point
        topo_mask = [clusters_mask[functions_general.find_nearest(tfr.freqs, timefreq[1] - 3)[0]:
                                   functions_general.find_nearest(tfr.freqs, timefreq[1] + 1)[0],
                     functions_general.find_nearest(tfr.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).times, timefreq[0] - 0.05)[0]:
                     functions_general.find_nearest(tfr.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).times, timefreq[0] + 0.05)[0]].
                     sum(axis=0).sum(axis=0).astype(bool) for timefreq in timefreqs]
        masks = []
        for topo in topo_mask:
            mask = np.zeros(len(tfr_topo.copy().pick('mag').info.ch_names)).astype(bool)
            mask[[idx for idx, channel in enumerate(tfr_topo.copy().pick('mag').info.ch_names) if channel in list(compress(tfr_plotjoint.info.ch_names, topo))]] = True
            masks.append(mask)
    else:
        masks = [None]*len(timefreqs)
    mask_params = dict(marker='o', markerfacecolor='white', markeredgecolor='k', linewidth=0, markersize=4, alpha=0.5)

    # Get topo axes and overwrite topoplots
    topo_axes = fig.axes[:-2]
    for i, (ax, timefreq) in enumerate(zip(topo_axes, timefreqs)):
        ax.clear()
        topomap_kw = dict(ch_type='mag', tmin=timefreq[0], tmax=timefreq[0], fmin=timefreq[1], fmax=timefreq[1], mask=masks[i], mask_params=mask_params, colorbar=False, show=display_figs)
        tfr_topo.plot_topomap(axes=ax, cmap=cmap, vlim=(vmin, vmax), **topomap_kw)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Get colorbar axis
    cbar_ax = fig.axes[-1]
    fig.colorbar(sm, cax=cbar_ax)

    if save_fig:
        save.fig(fig=fig, path=trf_fig_path, fname=fname)

    return fig, tf_ax

def mri_meg_alignment(subject, subject_code, dig, subjects_dir=os.path.join(paths.mri_path, 'FreeSurfer_out')):

    # Path to MRI <-> HEAD Transformation (Saved from coreg)
    trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
    fids_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-fiducials.fif')
    dig_info_path = paths.opt_path + subject.subject_id + '/info_raw.fif'

    # Load raw meg data with dig info
    info_raw = mne.io.read_raw_fif(dig_info_path)

    # Visualize MEG/MRI alignment
    surfaces = dict(brain=0.7, outer_skull=0.5, head=0.4)
    # Try plotting with head skin and brain
    try:
        mne.viz.plot_alignment(info_raw.info, trans=trans_path, subject=subject_code,
                                     subjects_dir=subjects_dir, surfaces=surfaces,
                                     show_axes=True, dig=dig, eeg=[], meg='sensors',
                                     coord_frame='meg', mri_fiducials=fids_path)
    # Plot only outer skin
    except:
        mne.viz.plot_alignment(info_raw.info, trans=trans_path, subject=subject_code,
                                     subjects_dir=subjects_dir, surfaces='outer_skin',
                                     show_axes=True, dig=dig, eeg=[], meg='sensors',
                                     coord_frame='meg', mri_fiducials=fids_path)


def connectivity_circle(subject, labels, con, surf_vol, vmin=None, vmax=None, connectivity_method='pli', subject_code=None, display_figs=False, save_fig=False,
                        fig_path=None, fname=None, fontsize=None, ticksize=None):
    # Sanity check
    if save_fig and (not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    # Plot fonsize params
    if fontsize:
        matplotlib.rc({'font.size': fontsize})
    if ticksize:
        matplotlib.rc({'xtick.labelsize': ticksize})
        matplotlib.rc({'ytick.labelsize': ticksize})

    # Get colors for each label
    if surf_vol == 'surface':
        label_colors = [label.color for label in labels]
    elif surf_vol == 'volume':
        label_colors = labels[1]

    # Reorder the labels based on their location in the left hemi
    if surf_vol == 'surface':
        label_names = [label.name for label in labels]
        lh_labels = [name for name in label_names if name.endswith('lh')]
    elif surf_vol == 'volume':
        label_names = labels[0]
        lh_labels = [name for name in label_names if ('lh' in name or 'Left' in name)]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)

    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2])

    # Plot the graph using node colors from the FreeSurfer parcellation
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', subplot_kw=dict(polar=True))

    # Plot
    mne_connectivity.viz.plot_connectivity_circle(con, label_names, n_lines=50, vmin=vmin, vmax=vmax,
                                                  node_angles=node_angles, node_colors=label_colors,
                                                  title=f'All-to-All Connectivity ({connectivity_method})', ax=ax,
                                                  show=display_figs)
    fig.tight_layout()

    # Save
    if save_fig:
        if not fname:
            fname = f'{subject.subject_id}_circle'
        if subject_code == 'fsaverage':
            fname += '_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)


def connectome(subject, labels, adjacency_matrix, subject_code, save_fig=False, fig_path=None, fname='GA_connectome', connections_num=.99,
               node_size=10, node_color='k', linewidth=2, cmap=None, edge_vmin=None, edge_vmax=None):

    # Sanity check
    if save_fig and (not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    # Get parcelation labels positions
    label_names = [label.name for label in labels]
    # Get the y-location of the label
    label_xpos = list()
    label_ypos = list()
    label_zpos = list()
    for name in label_names:
        idx = label_names.index(name)
        label_xpos.append(np.mean(labels[idx].pos[:, 0]))
        label_ypos.append(np.mean(labels[idx].pos[:, 1]))
        label_zpos.append(np.mean(labels[idx].pos[:, 2]))

    # Make node position array and reescale
    nodes_pos = np.array([label_xpos, label_ypos, label_zpos]).transpose() * 1100

    # Plot only if at least 1 connection
    if abs(adjacency_matrix).sum() > 0:
        # Define edges to plot
        if connections_num >= 1:
            edge_threshold = sorted(np.sort(adjacency_matrix, axis=None))[-int(connections_num * 2)]
            if not edge_vmin:
                edge_vmin = edge_threshold
            if not edge_vmax:
                edge_vmax = np.sort(adjacency_matrix, axis=None)[-1]
            if edge_vmin < 0:
                cmap = 'bwr'
            else:
                cmap = 'Reds'

        elif connections_num < 1:
            edge_threshold = f'{connections_num*100}%'
            if not edge_vmax:
                edge_vmax = adjacency_matrix.max()

            if adjacency_matrix.min() < 0:
                if not edge_vmin:
                    edge_vmin = adjacency_matrix.min()
                cmap = 'bwr'
            else:
                if not edge_vmin:
                    edge_vmin = sorted(np.sort(adjacency_matrix, axis=None))[-int((1 - connections_num) * len(np.sort(adjacency_matrix, axis=None))**2)]  # set colorbar minimum as twice smaller than plot minimum
                cmap = 'Reds'

        # Plot connectome
        fig = plotting.plot_connectome(adjacency_matrix=adjacency_matrix, node_coords=nodes_pos, edge_threshold=edge_threshold, node_size=node_size, node_color=node_color,
                                       edge_vmin=edge_vmin, edge_vmax=edge_vmax, edge_cmap=cmap, edge_kwargs=dict(linewidth=linewidth), colorbar=True)

        # Save
        if save_fig:
            if not fname:
                fname = f'{subject.subject_id}_connectome'
            if subject_code == 'fsaverage' and 'fsaverage' not in fname:
                fname += '_fsaverage'
            save.fig(fig=fig, path=fig_path, fname=fname)


def plot_con_matrix(subject, labels, adjacency_matrix, subject_code, save_fig=False, fig_path=None, fname='GA_matrix'):

    matplotlib.use('TkAgg')

    # Sanity check
    if save_fig and (not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    label_names = [label.name for label in labels]
    # Get the y-location of the label
    label_ypos = list()
    for name in label_names:
        idx = label_names.index(name)
        label_ypos.append(np.mean(labels[idx].pos[:, 1]))

    # Make adjacency matrix sorted from frontal to posterior
    sort = np.argsort(label_ypos)  # Get sorted indexes based on regions anterior-posterior order
    # adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.transpose())  # Make symetric
    sorted_matrix = adjacency_matrix[sort[::-1]]  # Sort on one axis
    sorted_matrix = sorted_matrix[:, sort[::-1]]  # Apply same sort on second axis

    sorted_labels = [label_names[i] for i in sort][::-1]  # Get labels in sorted order

    vmin = np.sort(sorted_matrix.ravel())[len(label_ypos)]
    vmax = np.sort(sorted_matrix.ravel())[-1]

    fig = plt.figure(figsize=(8, 5))
    im = plt.imshow(sorted_matrix, vmin=vmin, vmax=vmax)
    fig.colorbar(im)

    ax = plt.gca()
    ax.set_yticklabels(sorted_labels)
    ax.set_xticklabels([])

    plt.suptitle('')

    # Save
    if save_fig:
        if not fname:
            fname = f'{subject.subject_id}_matrix'
        if subject_code == 'fsaverage' and 'fsaverage' not in fname:
            fname += '_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)

    return sorted_matrix


def connectivity_strength(subject, subject_code, con, src, labels, surf_vol, subjects_dir, save_fig, fig_path, fname, threshold=1):

    try:
        mne.viz.close_all_3d_figures()
    except:
        pass

    # Plot connectivity strength (connections from each region to other regions)
    degree = mne_connectivity.degree(connectivity=con, threshold_prop=threshold)
    if surf_vol == 'surface':
        stc = mne.labels_to_stc(labels, degree)
        stc = stc.in_label(mne.Label(src[0]["vertno"], hemi="lh") + mne.Label(src[1]["vertno"], hemi="rh"))
        hemi = 'split'
        views = ['lateral', 'med']

    if surf_vol == 'volume':
        stc = mne.VolSourceEstimate(degree, [src[0]["vertno"]], 0, 1, "bst_resting")
        hemi = 'both'
        views = 'dorsal'

    brain = stc.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, size=(1000, 500), clim=dict(kind="percent", lims=[50, 75, 95]), hemi=hemi, views=views)
    if save_fig:
        if not fname:
            fname = f'{subject.subject_id}_strength'
        if subject_code == 'fsaverage' and 'fsaverage' not in fname:
            fname += '_fsaverage'
        brain.save_image(filename=fig_path + fname + '.png')
        brain.save_image(filename=fig_path + '/svg/' + fname + '.pdf')


def sources(stc, src, subject, subjects_dir, initial_time, surf_vol, force_fsaverage, estimate_covariance, save_fig, fig_path, fname, surface='pial', hemi='split', views='lateral',
            alpha=0.75, mask_negatives=False, time_label='auto', save_vid=True, positive_cbar=None, clim=None):

    # Close all plot figures
    try:
        mne.viz.close_all_3d_figures()
        plt.close('all')
    except:
        pass

    # Convert view to list in case only 1 view as str
    if type(views) == str:
        views = [views]

    # Define clim
    if not clim:
        clim = {'kind': 'values', 'lims': ((abs(stc.data).max() - abs(stc.data).min()) / 1.5,
                                           (abs(stc.data).max() - abs(stc.data).min()) / 1.25,
                                           (abs(stc.data).max() - abs(stc.data).min()))}

        # Replace positive cbar for positive / negative
        if positive_cbar == False or (stc.data.mean() - stc.data.std() <= 0 and positive_cbar != True):
            clim['pos_lims'] = clim.pop('lims')

    if surf_vol == 'volume':

        # Nutmeg plot
        if subject == 'fsaverage':
            bg_img = paths.mri_path + 'MNI_templates/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'
        else:
            bg_img = None

        # Set backend
        matplotlib.use('TkAgg')

        fig = stc.plot(src=src, subject=subject, subjects_dir=subjects_dir, initial_time=initial_time, clim=clim, bg_img=bg_img)
        if save_fig:
            if force_fsaverage:
                fname += '_fsaverage'
            if mask_negatives:
                fname += '_masked'
            os.makedirs(fig_path, exist_ok=True)
            save.fig(fig=fig, path=fig_path, fname=fname)

        # 3D plot
        brain = stc.plot_3d(src=src, subject=subject, subjects_dir=subjects_dir, hemi=hemi, views=views, clim=clim,
                            initial_time=initial_time, size=(1000, 500), time_label=time_label, brain_kwargs=dict(surf=surface, alpha=alpha))

        if save_fig:
            view_fname = fname + f'_3D'
            if force_fsaverage:
                view_fname += '_fsaverage'
            if mask_negatives:
                view_fname += '_masked'
            os.makedirs(fig_path + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path + view_fname + '.png')
            brain.save_image(filename=fig_path + '/svg/' + view_fname + '.pdf')
            if save_vid and not estimate_covariance:
                try:
                    brain.save_movie(filename=fig_path + view_fname + '.mp4', time_dilation=12, framerate=30)
                except:
                    pass

    # 3D plot
    elif surf_vol == 'surface':

        brain = stc.plot(src=src, subject=subject, subjects_dir=subjects_dir, hemi=hemi, clim=clim, initial_time=initial_time, views=views, size=(1000, 500),
                         brain_kwargs=dict(surf=surface, alpha=alpha))

        if save_fig:
            view_fname = fname + f'_3D'
            if force_fsaverage:
                view_fname += '_fsaverage'
            if mask_negatives:
                view_fname += '_masked'
            os.makedirs(fig_path + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path + view_fname + '.png')
            brain.save_image(filename=fig_path + '/svg/' + view_fname + '.pdf')
            if save_vid and not estimate_covariance:
                try:
                    brain.save_movie(filename=fig_path + view_fname + '.mp4', time_dilation=12, framerate=30)
                except:
                    pass

    return brain