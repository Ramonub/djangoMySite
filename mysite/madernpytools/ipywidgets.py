import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import madernpytools.frequency_response as frf
import madernpytools.plot as mplt


def plot_EMA_window_settings(window_fraction, overlap_fraction,
                             filename, fdir,
                             signal_settings=None, xlim=None, **kwargs):
    """Plot the frequency analysis for given window_fraction and overlap fraction
    params:
    window_fraction  : Size of window as fraction of number of samples
    overlap_fraction : Overlap of window as fraction of window size
    filename         : File name of the data file to analyze (use regular expression syntax to enable multiple files), e.g:
                'EMA_infeed_pull_R.*\.csv'
    dir              : Directory in which the data files are located
    signal_settings  : {'i_input':2, 'i_output': 0}
    """
    if signal_settings is None:
        signal_settings = {'i_input': 2, 'i_output': 0}

    xlim = [0, 1500] if xlim is None else xlim
    fig = plt.figure(figsize=(10, 5))

    # Plot results
    ma = frf.ModalAnalysis()
    ma.load_from_files(filename='{0}/{1}'.format(fdir, filename) , **signal_settings,
                       window_fraction=window_fraction, overlap_fraction=overlap_fraction,
                       f_range=xlim,
                       **kwargs
                       )

    mplt.SpectralPlot(figure=fig, plot_samples=True, xlim=xlim, spectral_data=ma.spectral_data)


def interact_EMA_window_settings(filename, dir='./data', **kwargs):
    """Show interactive widget which allows one to evaluate the Hann window settings: window fraction and overlap fraction.

    Window fraction indictes the fraction of the total number of samples which forms one window.
    The overlap fraction indicates the amount of overlap between windows

    params:
    filename: File name of the data file to analyze (use regular expression syntax to enable multiple files), e.g:
                'EMA_infeed_pull_R.*\.csv'

    optional params:
    dir : Directory in which the data files are located
    signal_settings: {'i_input':2, 'i_output': 0}
    """
    sl_window = widgets.FloatSlider(max=1.0, min=0, value=0.5, continous_update=False)
    sl_overlap = widgets.FloatSlider(max=1.0, min=0, value=0.5, continous_update=False)

    f_interact = lambda window_size, overlap_fraction: plot_EMA_window_settings(window_size, overlap_fraction,
                                                                                filename, dir, **kwargs)
    widgets.interact(f_interact, window_size=sl_window, overlap_fraction=sl_overlap)


def visualize_mode(modes, modal_analysis,
                   freq_index, save=False, **kwargs):
    """ Visualizes the eigen-frequencies and the mode of a selected frequency.
    :param modes         : ModeList() object
    :type modes          : frf.ModeList
    :param modal_analysis: ModalAnalysis object which corresponds to modelist
    :param freq_index    : Index of mode to display (starts at lowest frequency)
    :param save          : Flag which indicates to save current plot
    """
    f_sel = modes[0].get_frequencies()[freq_index]
    xlim = kwargs.get('xlim', [0, 2000])
    ylim = kwargs.get('ylim', [-1.1, 1.1])
    plot_keys = kwargs.get('plot_keys', 'H')

    # Create plot
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(211)
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_xlabel('Frequency  [Hz]')
    ax1.set_xlim(xlim)
    ax2 = plt.subplot(212)
    ax2.set_ylabel('Im(H)/max(Im(H)) [-]')
    ax2.set_xlabel('Cylinder location [mm]')
    ax3 = ax1.twinx()
    ax3.set_ylabel('Coherence [-]')

    # Plot spectral information:
    axs = {'magnitude': ax1, 'coherence': ax3}
    myplt = modal_analysis.plot_FRF(axs={'coherence': ax3, 'magnitude': ax1}, xlim=xlim,
                                    plot_keys=plot_keys)
    myplt.show_eigen_frequencies(freqs=np.array(modes[0].get_frequencies()))
    axs['coherence'].plot([f_sel, f_sel], [0, 1], '--', linewidth=3, color='purple')
    myplt.grid(True)

    for m in modes:
        freq = m.natural_frequency
        ind = np.where(freq <= f_sel)[0][-1]
        mplt.plot_mode(ax2, m[freq[ind]])
    ax2.set_ylim(ylim)
    ax2.set_title('Frequency: {0} [Hz]'.format(np.round(f_sel)))

    # plt.tight_layout()
    if save:
        plt.savefig('Mode{0}Hz.pdf'.format(f_sel), bbox_inches='tight')


def mode_interactive(modes, modal_analysis, **kwargs):
    """ Plot
    :param modes: ModalList object
    :param modal_analysis: A modal analysis used to generate the modes
    """
    if type(modes) is not list:
        modes = [modes]
    eig_freqs = kwargs.get('freqs', modes[0].get_frequencies())

    # Widgets:
    int_slider = widgets.IntSlider(value=5, min=0, max=len(eig_freqs), step=1, description='Frequency [Hz]',
                                   continuous_update=False, readout=True, readout_format='d')
    toggle_btn = widgets.ToggleButton(value=False, description='Save')

    # Define function:
    f_interact = lambda freq, save: visualize_mode(modes, modal_analysis,
                                                         freq_index=freq, save=save, **kwargs)
    widgets.interact(f_interact, freq=int_slider, save=toggle_btn)

