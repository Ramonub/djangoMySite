import os, copy, re
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from tkinter import messagebox as mbox
from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import madernpytools.backbone as mbb
import madernpytools.signal_handling as msignals
import madernpytools.backbone as bb
import madernpytools.tools.frequency_response as mfrf

from madernpytools.backbone import *
from madernpytools.signal_handling import Signal
from madernpytools.signal_handling import SignalSubscriber, SignalList
from madernpytools.log import LogInfo, AutoSaveLog


class AutoRefreshFrame(tk.Frame):

    def __init__(self, master=None, cb_attr='refresh'):
        tk.Frame.__init__(self, master=master)
        self._refresh_rate = 0.0
        self._refresh_time = 0.0
        self._auto_refresh = True
        self._refresh_attr = cb_attr

        # Initialize values:
        self.refresh_rate = 60

    def refresh(self):
        raise NotImplementedError()

    @property
    def refresh_rate(self):
        """Auto refresh rate"""
        return self._refresh_rate

    @refresh_rate.setter
    def refresh_rate(self, value):
        """Auto refresh rate"""
        self._refresh_rate = value
        self._refresh_time = int(1.0 / value * 1e3)

    @property
    def auto_refresh(self):
        return self._auto_refresh

    @auto_refresh.setter
    def auto_refresh(self, value):
        self._auto_refresh = value
        if value:
            self.after(self._refresh_time, self._cb_refresh)

    def _cb_refresh(self):
        """Private function, internally called on a refresh call
        """
        if self._auto_refresh:
            # Call refresh attribute:
            getattr(self, self._refresh_attr)()

            # Activate next call
            self.after(self._refresh_time, self._cb_refresh)


class FileBrowser(ttk.Frame):

    def __init__(self, parent, **kwargs):
        ttk.Frame.__init__(self, master=parent)

        self._dir = LabeledEntry(self, label='Directory', entry_conf={'width': 50})
        self._dir.value = kwargs.get('directory', os.path.expanduser('~'))
        self._dir.grid(column=0, row=0, sticky=tk.E)

        self._dir_btn = tk.Button(self, text='...', command=self._cb_browsedir)
        self._dir_btn.grid(column=1, row=0, sticky=tk.W)

        self._filename = LabeledEntry(self, label='Filename', entry_conf={'width': 50})
        self._filename.value = kwargs.get('filename', 'log.csv')
        self._filename.grid(column=0, row=1, sticky=tk.E)

        self._fnbtn = tk.Button(self, text='...', command=self._cb_browsefile)
        self._fnbtn.grid(column=1, row=1, sticky=tk.W)

    def _cb_browsedir(self):
        dir_name = tk.filedialog.askdirectory(initialdir=self._dir.value)
        self._dir.value = dir_name

    def _cb_browsefile(self):
        fn = tk.filedialog.askopenfilename(initialfile=self._filename.value)
        self._filename.value = fn

    @property
    def full_filename(self):
        return '{0}/{1}'.format(self._dir.value, self._filename.value)

    @property
    def directory(self):
        return '{0}'.format(self._dir.value, self._filename.value)

    @property
    def filename(self):
        return '{1}'.format(self._dir.value, self._filename.value)

    def config(self, **kwargs):
        self._fnbtn.config(**kwargs)
        self._filename.config(**kwargs)
        self._dir.config(**kwargs)
        self._dir_btn.config(**kwargs)

    def existing_files(self, pattern='{0}.csv'):
        sel_files = [f for f in os.listdir(self.directory) if re.match(pattern.format(
            os.path.splitext(self.filename)[0]), f)]
        return sel_files


class LogWidget(tk.Frame):
    def __init__(self, parent=None, record_command=None, stop_command=None):
        """Widget with filebrowser, 'record' and 'stop' button. """
        tk.Frame.__init__(self, parent)
        self.grid()

        # Filebrowser
        self.file_browser = FileBrowser(self)
        self.file_browser.grid(column=1, row=0, rowspan=2, sticky=tk.W)

        # Log description
        self._description = LabeledEntry(self, label='Description', entry_conf={'width': 50})
        self._description.grid(column=1, row=3, rowspan=2, sticky=tk.W)

        # Buttons:
        self._btn_record = tk.Button(self, text='Record', command=self._cb_record, width=6)
        self._btn_record.grid(column=0, row=0, sticky=tk.W)
        self._btn_stop = tk.Button(self, text='Stop', command=self._cb_stop, width=6)
        self._btn_stop.grid(column=0, row=1, sticky=tk.W)

        self._record_cmd = record_command
        self._stop_cmd = stop_command

        # Events:
        self.event_record = mbb.EventPublisher(str)
        self.event_stop = mbb.EventPublisher(str)

    def _check_filename(self):
        files = self.file_browser.existing_files(pattern='{0}\d{{6}}.csv')
        all_ok = True

        if len(files) > 0:
            # Create message_lbl:
            mes = "Files with selected filename already exist: \n{0}do you want to overwrite them?".format(
                ''.join(' - {0},\n'.format(f) for f in files))
            if not mbox.askyesno("File(s) exist", message=mes, icon=mbox.WARNING, parent=self):
                # Abort
                all_ok = False

        return all_ok

    def _cb_record(self):
        """ Internal callback for record button

        :return:
        """
        if self._check_filename():
            # Set qtgui state:
            self._btn_record.config(relief=tk.SUNKEN)
            self.file_browser.config(state=tk.DISABLED)
            self._description.config(state=tk.DISABLED)

            # Notify subscribers:
            self._record_cmd() if self._record_cmd is not None else None
            self.event_record.raise_event(self.file_browser.full_filename)

    def _cb_stop(self):
        """ Internal callback for stop button

        :return:
        """
        self.event_stop.raise_event(self.file_browser.full_filename)
        self._stop_cmd() if self._stop_cmd is not None else None

        self._btn_record.config(relief=tk.RAISED)
        self.file_browser.config(state=tk.NORMAL)
        self._description.config(state=tk.NORMAL)

    @property
    def description(self):
        return self._description.value

    @description.setter
    def description(self, value):
        self._description.value


class SignalValueIndicator(AutoRefreshFrame):

    def __init__(self, parent, signal: msignals.Signal):
        """"Signal value indicator"""
        AutoRefreshFrame.__init__(self, master=parent)

        self.event_clicked = mbb.EventPublisher(data_type=SignalValueIndicator)
        self._signal_sub = mbb.SimpleEventSubscriber(self._new_value)
        self._n_decimals = 2

        self._signal = signal
        self._signal_var = tk.StringVar()
        self._signal_var.set(' 0.00')
        self.name = signal.name

        # Define labels:
        self._signal_lbl = tk.Label(self, textvariable=self._signal_var, text=self.name, relief=tk.SUNKEN)
        self._signal_lbl.grid(column=1, row=0)

        self._name_lbl = tk.Label(self, text=self._signal.name)
        self._unit_lbl = tk.Label(self, text=self._signal.unit)

        self._signal.connect(self._signal_sub)

        # Define callback:
        self._signal_lbl.bind("<Double-Button-1>", self._value_click)

        self.refresh_rate = 5
        self.auto_refresh = True

    @property
    def value(self):
        """Signal value"""
        if self._signal.value is None:
            return -1
        else:
            return float(self._signal.value)

    def _val2str(self, value):
        """"Convert value into formated string"""
        return '{{0: .{0}f}}'.format(self._n_decimals).format(value)

    def _value_click(self, e):
        """"Response to click on value"""
        self.event_clicked.raise_event(self)

    def _new_value(self, publisher):
        # Signal value changed
        self._value = publisher.value

    def refresh(self):
        """"Refresh displayed value"""
        self._signal_var.set(self._val2str(self.value))

    @property
    def signal(self):
        """"Signal whose value is displayed by widget."""
        return self._signal

    def show_name(self, value):
        if value:
            self._name_lbl.grid(column=0, row=0)

    def show_unit(self, value):
        if value:
            self._unit_lbl.grid(column=2, row=0)


class SignalProperties(tk.Frame):

    def __init__(self, signal: msignals.Signal, parent=None):
        # Initialize frame with grid layout
        tk.Frame.__init__(self, parent)
        self.grid()

        # Define items to display
        settings = [{'name': 'Name', 'var_type': tk.StringVar, 'attr': 'name'},
                    {'name': 'Unit', 'var_type': tk.StringVar, 'attr': 'unit'},
                    {'name': 'Module', 'var_type': tk.StringVar, 'attr': 'module'},
                    {'name': 'Channel', 'var_type': tk.IntVar, 'attr': 'channel_index'},
                    {'name': 'Calibration value', 'var_type': tk.StringVar, 'attr': 'calibration_value'}
                    ]

        # Create items:
        self._labels = {}
        for i, item in enumerate(settings):
            # Label:
            lbl_name = tk.Label(self, text='{0}\t'.format(item['name']))
            lbl_name.grid(column=0, row=i, sticky=tk.W)

            # Variable
            var = item['var_type']()
            var.set(getattr(signal, item['attr']))

            # Value display
            lbl_val = tk.Entry(self, textvariable=var)
            lbl_val.grid(column=1, row=i, sticky=tk.W)

            # Add to dictionary of labels:
            self._labels[item['name']] = {'name_label': tk.Label(text=item['name']),
                                          'variable': var,
                                          'value_label': lbl_val
                                          }


class AbstractLine(EventSubscriber):

    def update(self, publisher):
        raise NotImplementedError()


class SignalLine(object):

    def __init__(self, ax, signal_list, line_spec=None):
        # Verify arguments
        line_spec = ArgumentVerifier(dict, {}).verify(line_spec)
        signal_list = ArgumentVerifier(SignalList, None).verify(signal_list)

        # Define line:
        self._l, = ax.plot([], [], label=r'{0}[${1}$]'.format(signal_list[1].name, signal_list[1].unit), **line_spec)
        self._signal_list = signal_list

        self._sub_new_data = SimpleEventSubscriber(self._data_update)
        self._signal_list.connect(self._sub_new_data)
        self._data = None

    def set_data(self, data: np.ndarray):
        self._data = data

    def _data_update(self, publisher: SignalList):
        self._data = [publisher[0].data,  # X-data
                      publisher[1].data]  # Y-data

    def refresh(self):
        if self._data is not None:
            self._l.set_data(self._data[0],
                             self._data[1])

    @property
    def line(self):
        return self._l

    def remove(self):
        self._l.remove()
        del self._l


class ItemCheckBox(tk.Frame):

    def __init__(self, item, parent):
        tk.Frame.__init__(self, master=parent)
        self.event_checked = mbb.EventPublisher(data_type=ItemCheckBox)
        self.item = item

        # Define callback:
        self._var = tk.IntVar()
        self._signal_chk = tk.Checkbutton(self, text=item, command=self._value_click, variable=self._var)
        self._signal_chk.pack()

    @property
    def state(self):
        return self._var.get()

    def _value_click(self):
        self.event_checked.raise_event(self)


class SignalCheckbox(tk.Frame):

    def __init__(self, signal, parent):
        tk.Frame.__init__(self, master=parent)
        self.event_checked = mbb.EventPublisher(data_type=SignalCheckbox)
        self.signal = signal

        # Define callback:
        self._var = tk.IntVar()
        self._signal_chk = tk.Checkbutton(self, text=signal.name, command=self._value_click, variable=self._var)
        self._signal_chk.pack()

    @property
    def state(self):
        return self._var.get()

    @property
    def name(self):
        return self.signal.name

    def _value_click(self):
        self.event_checked.raise_event(self)


class SignalSelection(tk.Frame):

    def __init__(self, master=None, signals: msignals.SignalDict = None):
        tk.Frame.__init__(self, master=master)
        self.grid()

        signals = ArgumentVerifier(msignals.SignalDict, msignals.SignalDict()).verify(signals)

        self.event_checked = mbb.EventPublisher(data_type=SignalCheckbox)
        self._click_subscriber = mbb.SimpleEventSubscriber(self._cb_clicked)
        self.check_boxes = []

        i = 0
        for key, s in signals.items():
            val = SignalCheckbox(s, self)
            val.event_checked.connect(self._click_subscriber)
            val.grid(column=0, row=i, sticky=tk.W)
            self.check_boxes.append(val)
            i += 1

    def _cb_clicked(self, publisher):
        widget = publisher.get_data()
        self.event_checked.raise_event(publisher.get_data())


class GUIPlot(AutoRefreshFrame):

    def __init__(self, master=None, display_nav_bar=False, **kwargs):
        AutoRefreshFrame.__init__(self, master=master)

        # Setup figure:
        self._fig = Figure(tight_layout=True, facecolor=None)
        self._fig.set_figheight(kwargs.get('height', 300) / self._fig.dpi)
        self._fig.set_figwidth(kwargs.get('width', 400) / self._fig.dpi)

        # Setup canvas:
        self._canvas = FigureCanvasTkAgg(figure=self._fig, master=self)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        if display_nav_bar:
            NavigationToolbar2Tk(self._canvas, self)

    def refresh(self):
        self._canvas.draw_idle()

    @property
    def fig(self):
        return self._fig


class SignalPlot(AutoRefreshFrame):

    def __init__(self, master, line_properties=None, **kwargs):
        AutoRefreshFrame.__init__(self, master=master)

        # Variables
        self._line_properties = ArgumentVerifier(dict, {}).verify(line_properties)
        self._window = kwargs.pop('window', 1000)
        self._show_legend = True

        # Setup figure:
        self._fig = Figure(tight_layout=True, facecolor=None)
        self._fig.set_figheight(kwargs.get('height', 300) / self._fig.dpi)
        self._fig.set_figwidth(kwargs.get('width', 400) / self._fig.dpi)
        self._ax = self._fig.gca()

        # Setup canvas:
        self._canvas = FigureCanvasTkAgg(figure=self._fig, master=self)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self._relim = 'xy'

        self._lines = {}
        self.auto_refresh = True

    def append(self, signal_pair):
        if not str(signal_pair.name) in self._lines:
            self._lines[str(signal_pair.name)] = SignalLine(self._ax, signal_pair, self._line_properties)
            # Refresh legend
            self._update_legend()
        else:
            print('Signal already displayed')

    def remove(self, signal_pair):
        if str(signal_pair.name) in self._lines:
            self._lines.pop(str(signal_pair.name)).remove()

            # Refresh legend
            self._update_legend()

    def _update_legend(self):
        if self._show_legend and len(self._lines) > 0:
            self._ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), fontsize=8,
                            loc=3, ncol=4, mode='expand', borderaxespad=0.)  # Refresh legend()
        elif self._ax.legend() is not None:
            self._ax.legend().set_visible(False)

    def legend(self, show=True):
        self._show_legend = show

    def refresh(self):

        for key, l in self._lines.items():
            l.refresh()

        if self._relim != '':
            x_lim = self._ax.get_xlim()
            y_lim = self._ax.get_ylim()

            self._ax.relim()
            if not ('x' in self._relim):
                self._ax.set_xlim(x_lim)
            if not ('y' in self._relim):
                self._ax.set_ylim(y_lim)
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax


class TimePlot(tk.Frame):

    def __init__(self, parent, signal_dict, **kwargs):
        tk.Frame.__init__(self, master=parent)

        self.grid()
        self._sub_signal_update = mbb.SimpleEventSubscriber(self._signal_update)

        # Generate signal pairs:
        time_signal = signal_dict['time']
        self._signal_pairs = msignals.SignalDict()
        self._selection = msignals.SignalDict()
        for key, s in signal_dict.items():
            if s is not time_signal:
                # Create new pair, and use s as identifier:
                tmp_sigpair = SignalList([time_signal, s])
                self._signal_pairs[str(s.name)] = tmp_sigpair
                self._selection[str(s.name)] = s

        # GUI Elements:
        self._item_selection = SignalSelection(self, signals=self._selection)
        self._item_selection.grid(column=0, row=0)

        # Create plot (supply empty list)
        self._plot = SignalPlot(self, **kwargs)
        self._plot.ax.set_xlabel('Time [s]')
        self._plot.grid(column=1, row=0)
        self._plot.legend()

        # Event & Subscribers:
        self._check_subscriber = SimpleEventSubscriber(h_callback=self._cb_checked)
        self._item_selection.event_checked.connect(self._check_subscriber)
        signal_dict.connect(self._sub_signal_update)

    def _cb_checked(self, publisher):

        # Publisher data is SignalCheckbutton
        checked_signal = str(publisher.get_data().signal.name)  # Get signal
        state = publisher.get_data().state  # Get signal

        # Respond to check:
        if state:
            self._plot.append(self._signal_pairs[checked_signal])
        else:
            self._plot.remove(self._signal_pairs[checked_signal])

    @property
    def ax(self):
        return self._plot.ax

    def _signal_update(self, publisher):
        # Notify signal list users, that the signal list values have changed
        for _, signal_list in self._signal_pairs.items():
            signal_list.raise_event(signal_list)

    def set_auto_refresh(self, value=True):
        self._plot.auto_refresh = value

    @property
    def plot(self):
        return self._plot


class DictSetter(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, master=parent)
        self.grid()

        self._dictionary = {}

        # Items
        self._cb_value = tk.StringVar()
        self._cb_box = ttk.Combobox(self, textvariable=self._cb_value)
        self._cb_box.grid(column=0, row=0, sticky=tk.W)
        self._cb_box.state(['!disabled', 'readonly'])
        self._cb_box.bind('<<ComboboxSelected>>', self._new_selection)

        self._value = tk.StringVar()
        self._entryValue = ttk.Entry(self, textvariable=self._value)
        self._entryValue.grid(column=1, row=0, sticky=tk.E)

        self._btn = ttk.Button(self, text='Update', command=self._update)
        self._btn.grid(column=2, row=0, sticky=tk.W)

        # Events:
        self._event_changed = EventPublisher(data_type=DictSetter)

    @property
    def event_changed(self):
        return self._event_changed

    @property
    def dictionary(self):
        return self._dictionary

    @dictionary.setter
    def dictionary(self, value):
        self._dictionary = value
        self._load_combobox(list(value.keys()))

    def _new_selection(self, _):
        print('Change detected')
        cb_item = self._cb_value.get()
        if cb_item in self._dictionary.keys():
            self._value.set(self._dictionary[cb_item])
        else:
            self._value.set('')

    def _load_combobox(self, items):
        self._cb_box['values'] = tuple(items)
        self._cb_box.current(0)
        self._new_selection(None)

    def _update(self):

        print('Update detected')
        # Update current value of combobox:
        if self._cb_value.get() in self._dictionary.keys():
            # Cast:
            item_type = type(self._dictionary[self._cb_value.get()])
            self._dictionary[self._cb_box.get()] = item_type(self._value.get())
            self._event_changed.raise_event(self)


class SpectralConfigurator(tk.Frame):

    def __init__(self, master=None, defaults=None):
        tk.Frame.__init__(self, master=master)
        self.grid()

        self.event_changed = EventPublisher(data_type=dict)

        self._change_subscriber = SimpleEventSubscriber(self._value_changed)

        #
        self._lbl_nperseg = LabeledEntry(self, label='n per segment', unit=' ',
                                         textvariable=tk.IntVar())
        self._lbl_nperseg.event_value_changed.connect(self._change_subscriber)

        #
        self._lbl_noverlap = LabeledEntry(self, label='n overlap', unit=' ',
                                          textvariable=tk.IntVar())
        self._lbl_noverlap.event_value_changed.connect(self._change_subscriber)

        #
        self._lbl_window_size = LabeledEntry(self, label='window size', unit='-',
                                             textvariable=tk.IntVar())
        self._lbl_window_size.event_value_changed.connect(self._change_subscriber)

    def _value_changed(self, publisher):
        self.event_changed.raise_event({'nperseg': self._lbl_nperseg.value,
                                        'noverlap': self._lbl_noverlap.value,
                                        }
                                       )


class SpectralPlot(tk.Frame):

    def __init__(self, parent=None, signal_dict=None, **kwargs):
        tk.Frame.__init__(self, master=parent)

        signal_dict = ArgumentVerifier(msignals.SignalDict, msignals.SignalDict()).verify(signal_dict)

        # Setup spectral analyzer
        self._setup_spectral_analysis(signal_dict, kwargs.get('spectral_settings', None))

        # GUI Elements:
        self._item_selection = SignalSelection(self, signals=self._time_signals)
        self._item_selection.grid(column=0, row=0)

        # Create plot (supply empty list)
        self._plot = SignalPlot(self, **kwargs)
        self._plot.grid(column=1, row=0)
        self._plot.ax.set_xlabel('Frequency [Hz]')
        self._plot.ax.set_ylabel('Magnitude [dB]')
        self._plot.ax.grid()
        self._plot.legend()

        # Event & Subscribers:
        self._check_subscriber = SimpleEventSubscriber(h_callback=self._cb_checked)
        self._item_selection.event_checked.connect(self._check_subscriber)

    def _config_changed(self, publisher):
        config = publisher.get_data()  # Get data from configuration

        for key, item in config.items():
            if key in self._spectral_analyzer.configuration.csd_args:
                self._spectral_analyzer.configuration.csd_args[key] = item

    def _cb_checked(self, publisher):

        # Publisher data is SignalCheckbutton
        checked_signal = str(publisher.get_data().signal.name)  # Get signal
        state = publisher.get_data().state  # Get signal

        # Respond to check:
        if state:
            self._plot.append(self._spectral_signals[checked_signal])
        else:
            self._plot.remove(self._spectral_signals[checked_signal])

    def _setup_spectral_analysis(self, signal_dict, spectral_settings=None):

        # Data locks:
        self._signal_lock = thrd.Lock()
        self._spectral_lock = thrd.Lock()
        self._spectral_analyzer_lock = thrd.Lock()

        # Main thread data
        self._time_signals = copy.copy(signal_dict)
        self._time_signals.pop('time')
        self._spectral_signals = msignals.SignalDict()
        for key, _ in self._time_signals.items():
            s_freq = Signal(name='frequency', data_index=0, unit='Hz')
            s_resp = Signal(name='{0}'.format(key), data_index=1, unit='dB')
            self._spectral_signals[key] = SignalList()
            self._spectral_signals[key].append(s_freq)
            self._spectral_signals[key].append(s_resp)

        # Analyzer thread
        self._bg_spectral_data = {}  # Thread memory for output data
        self._bg_time_signals = {}  # Thread memory for output data
        for key, s in self._time_signals.items():
            #if isinstance(s, Signal):
            self._bg_spectral_data[key] = np.array([[0.0, 0.0]])
            self._bg_time_signals[key] = Signal(key, signal_dict[key].unit, -1) #signal_dict[key].data_index)

        # Define analyser
        if spectral_settings is not None:
            spectral_config = mfrf.SpectralAnalyzerConfiguration.from_xml(spectral_settings)
        else:
            spectral_config = mfrf.SpectralAnalyzerConfiguration()
        self._spectral_analyzer = mfrf.SpectralAnalyzer(spectral_config)

        # Setup separate thread:
        self._spectral_job = bb.RepetitiveJob(rate=10)
        self._spectral_job.work = self._spectral_analysis
        self._bg_worker = bb.BackgroundWorker(self._spectral_job)

    def start(self):
        self._bg_worker.start()

    def stop(self):
        self._bg_worker.stop()

    def set_auto_refresh(self, value=True):
        if value:
            self.start()
        else:
            self.stop()
        self._plot.auto_refresh = value

    @property
    def spectral_analyzer(self) -> mfrf.SpectralAnalyzer:
        return self._spectral_analyzer

    @property
    def ax(self):
        return self._plot.ax

    @property
    def plot(self):
        return self._plot

    def _spectral_analysis(self):

        data_available = True
        # Copy data into thread memory:
        with self._signal_lock:
            for key, signal in self._time_signals.items():
                if not (signal.data is None):
                    self._bg_time_signals[key].data = copy.copy(signal.data)
                else:
                    data_available = False
                    break

        self._spectral_analyzer.configuration.i_output = 0
        self._spectral_analyzer.configuration.i_input = None

        if data_available:
            # Perform spectral analysis
            with self._spectral_analyzer_lock:
                for key, signal in self._bg_time_signals.items():
                    if signal.data.shape[0] > 200:
                        # Analyze data corresponding to signal s
                        spectral_sample = self._spectral_analyzer.analyze(signal.data[:, None])

                        # Save results:
                        if (spectral_sample.Sxx > 0.0).astype(int).sum() == len(spectral_sample.Sxx):
                            self._bg_spectral_data[key] = np.hstack([spectral_sample.frequencies[:, None],
                                                                     20 * np.log10(np.abs(spectral_sample.Sxx))[:, None]
                                                                     ])

            # Copy results into plot data memory:
            with self._spectral_lock:
                for key, spectral_data in self._bg_spectral_data.items():
                    # Update of this spectral data:
                    self._spectral_signals[key][0].data = spectral_data[:, 0]
                    self._spectral_signals[key][1].data = spectral_data[:, 1]

                    # Notify subscriber about update
                    self._spectral_signals[key].raise_event(self._spectral_signals[key])


class LabeledSignal(object):

    def __init__(self, parent, signal, row):
        # Create label
        self._label = tk.Label(parent, text=signal.name)
        self._label.grid(column=0, row=row, sticky='W')

        # Create variable
        self._variable = tk.DoubleVar()
        self._variable.set(signal.value)
        self._value = tk.Label(parent, textvariable=self._variable, width=10, anchor=tk.S)
        self._value.grid(column=1, row=row, sticky='E')

        # Create unit
        self._unit = tk.Label(parent, text='[{0}]'.format(signal.unit))
        self._unit.grid(column=2, row=row, sticky='W')
        self._n_decimals = 2
        self.data = signal.value

        self._subscriber = SignalSubscriber(h_callback=self._cb_signal_update)
        signal.connect(self._subscriber)

    def refresh(self):
        value = self.data
        if value is not None:
            self._variable.set(np.round(value, self._n_decimals))

    def _cb_signal_update(self, publisher):
        self.data = publisher.get_data().value


class SignalListWidget(AutoRefreshFrame):

    def __init__(self, parent, **kwargs):
        """Widget to display channel values"""
        AutoRefreshFrame.__init__(self, master=parent)

        self._frame = tk.LabelFrame(parent, text='Sensor values', **kwargs)
        self._labeled_signals = {}
        self.auto_refresh = True

    def _append(self, signal):

        if not (signal in self._labeled_signals):
            # add
            row = len(self._labeled_signals)
            self._labeled_signals[signal] = LabeledSignal(self._frame, signal, row=row)
            self._labeled_signals[signal].data = signal.value
        else:
            print('Signal already added')

    def _remove(self, signal):
        if signal in self._labeled_signals:
            self._labeled_signals.pop(signal)

    def refresh(self):
        for s, item in self._labeled_signals.items():
            item.refresh()

    def grid(self, *args, **kwargs):
        self._frame.grid(*args, **kwargs)


class LabeledItem(tk.Frame):

    def __init__(self, master=None, width=200):
        tk.Frame.__init__(self, master=master, width=width)
        self.event_value_changed = mbb.EventPublisher(data_type=LabeledItem)
        self.grid()
        self.grid_columnconfigure(index=0, minsize=100)

    @property
    def value(self):
        raise NotImplementedError()

    @value.setter
    def value(self, value):
        raise NotImplementedError()

    def config(self, **kwargs):
        raise NotImplementedError()


class LabeledOption(LabeledItem):

    def __init__(self, master, options, label='', unit='', default_value=None, **kwargs):
        LabeledItem.__init__(self, master=master)
        self.grid()

        default_value = options[0] if default_value is None else default_value

        self._update_time = kwargs.get('update_time', 500)

        self._label = tk.Label(self, text=label)
        self._label.grid(column=0, row=0, sticky='W')

        self._value = tk.StringVar()
        self._value.set(default_value)

        self._value_option = tk.OptionMenu(self, self._value, *options)
        self._value_option.grid(column=1, row=0, sticky='E')

        self._unit_label = tk.Label(self, text='[{0}]'.format(unit))
        self._unit_label.grid(column=2, row=0, sticky='W')

    @property
    def value(self):
        return self._value.get()

    @value.setter
    def value(self, value):
        self._value.set(value)


class LabeledValue(LabeledItem):

    def __init__(self, master=None, label='', unit='', default_value=0.0, **kwargs):
        LabeledItem.__init__(self, master=master)
        self.grid()
        self._update_time = kwargs.get('update_time', 500)

        self._label = tk.Label(self, text=label)
        self._label.grid(column=0, row=0, sticky='E')

        self._value = tk.StringVar()
        self._value.set(default_value)

        self._value_label = tk.Label(self, textvariable=self._value)
        self._value_label.grid(column=1, row=0, sticky='W')

        if unit != '':
            self._unit_label = ttk.Label(self, text='[{0}]'.format(unit))
        else:
            self._unit_label = ttk.Label(self, text=''.format(unit))
        self._unit_label.grid(column=2, row=0, sticky='W')

    @property
    def value(self):
        return self._value.get()

    @value.setter
    def value(self, value):
        self._value.set(value)

    def config(self, **kwargs):
        self._value_entry.config(**kwargs)


class ValidatingEntry(tk.Entry):
    # base class for validating entry widgets
    # Todo: Test

    def __init__(self, master, value="", value_format='.*'):
        tk.Entry.__init__, (self, master)

        self._value_format = value_format
        self.__value = value
        self.__variable = tk.StringVar()
        self.__variable.set(value)
        self.__variable.trace("w", self.__callback)
        self.config(textvariable=self.__variable)

    def __callback(self, *dummy):
        value = self.__variable.get()
        newvalue = self.validate(value)
        if newvalue is None:
            self.__variable.set(self.__value)
        elif newvalue != value:
            self.__value = newvalue
            self.__variable.set(self.newvalue)
        else:
            self.__value = value

    def validate(self, value):
        if re.match(self.value_format, value):
            return value

        # override: return value, new value, or None if invalid


class LabeledEntry(LabeledItem):

    def __init__(self, master=None, label='', unit='', value_format='.*', **kwargs):
        LabeledItem.__init__(self, master=master)
        self._label = ttk.Label(self, text='{0} '.format(label))
        self._label.grid(column=0, row=0, sticky='E')
        self.columnconfigure(0, minsize=100, pad=0)
        self._value_format = value_format

        self._value = kwargs.get('textvariable', tk.StringVar())
        self._value_entry = ttk.Entry(self, textvariable=self._value,
                                      **kwargs.get('entry_conf', dict()))
        self._value_entry.grid(column=1, row=0, sticky='W')
        self._value_entry.bind('<FocusOut>', self._entry_focus_out)

        if unit != '':
            self._unit_label = ttk.Label(self, text='[{0}]'.format(unit))
        else:
            self._unit_label = ttk.Label(self, text=''.format())
        self._unit_label.grid(column=2, row=0, sticky='W')

    def _entry_focus_out(self, e):
        self.event_value_changed.raise_event(self)

    @property
    def value(self):
        return self._value.get()

    @value.setter
    def value(self, value):
        self._value.set(value)

    @property
    def variable(self):
        return self._value

    def config(self, **kwargs):
        self._value_entry.config(**kwargs)


class SignalVisualizer(tk.Frame):

    def __init__(self, master=None, signals=None, display_settings: ET.ElementTree = None):
        """ The signal visualizer displays the signal values on an image. The object scans the provided display_settings
        for names that matches the signal names in the signal dictionary 'signals'. If the signal is found, it is visualized.

        The display configuration is determined by the
        display settings, an XML-tree which should have the following tree structure elements:
            * <image>
                * path: The drive location at which the image is located
            * <display_values>
                * <value name='signal name'>
                    * <show_name> : Boolean indicating if the signal name should be displayed
                    * <show_unit> : Boolean indicating if the signal unit should be displayed
                    * <rel_x>     : float [0-1] indicating the value x position relative to the image upper-left corner
                    * <rel_y>     : float [0-1] indicating the value y position relative to the image upper-left corner
                * <value name = 'another signal name'>
                    * ...

        :param master:  parent window in which the signal visualizer is placed
        :param signals: Signal dictionary  with signals
        :param display_settings: XML tree defining the location of the signals
        """

        # Set Frame
        tk.Frame.__init__(self, master=master, relief=tk.GROOVE)
        self.grid()

        # Publishers:
        self.event_signal_value_clicked = mbb.EventPublisher(SignalValueIndicator)

        # Internal subscribers:
        self._click_subscriber = mbb.SimpleEventSubscriber(self._value_click)

        # Load sensor values:
        self._sensor_labels = {}
        self._sensor_values = {}
        self._sensor_values = {}
        if display_settings is not None:

            # Set image as label:
            self.photo = SignalVisualizer.load_image(display_settings.find('image').find('path').text)

            self._imlabel = tk.Label(self, image=self.photo)
            self._imlabel.grid(column=0, row=0)
            self._n_decimals = 1

            for value in display_settings.find('display_values').iter('value'):
                name = value.get('name')
                if name in signals:
                    item = SignalValueIndicator(parent=self, signal=signals[name])
                    item.show_name(value.find('show_name').text == 'True')
                    item.show_unit(value.find('show_unit').text == 'True')
                    self._sensor_values[name] = item
                    self._sensor_values[name].place(relx=float(value.find('rel_x').text),
                                                    rely=float(value.find('rel_y').text),
                                                    anchor=tk.CENTER)
                    self._sensor_values[name].event_clicked.connect(self._click_subscriber)

        self._signals = signals

    @staticmethod
    def load_image(filename: str) -> ImageTk.PhotoImage:
        """ Load image of given filename

        :param filename:
        :type filename: str
        :return:
        :rtype: ImageTk.PhotoImage
        """
        with open(filename, 'rb') as im_handle:
            im = Image.open(im_handle)
            photo = ImageTk.PhotoImage(im)
            im.close()
            return photo

    def _value_click(self, publisher):
        """ Callback to handle signal value click events.

        :param publisher:
        :return:
        """
        newwindow = tk.Toplevel()
        newwindow.title("Signal properties")
        info = SignalProperties(self._signals[publisher.get_data().name], parent=newwindow)

    def set_auto_refresh(self, value=True):
        """ Defines autorefresh behavior

        :param value:
        :return:
        """
        for _, item in self._sensor_values.items():
            item.auto_refresh = value


class TestGUI(tk.Frame):

    def __init__(self, signal_dict, master=None):
        tk.Frame.__init__(self, master)
        self.grid()

        self._controls = LogWidget(self, record_command=self._cb_start_btn,
                                   stop_command=self._cb_stop_btn)
        self._controls.grid(column=1, row=1)

        self._nb = ttk.Notebook(self)
        self._nb.grid(column=1, row=0, columnspan=2)

        self._fig = TimePlot(self, signal_dict, height=300, width=500)
        self._nb.add(self._fig, text='time display')

        self._frf_fig = SpectralPlot(self, signal_dict=signal_dict, height=300, width=500)
        self._frf_fig.spectral_analyzer.configuration.sampling_rate = 10000
        self._frf_fig.spectral_analyzer.configuration.window_fraction = 0.2
        self._frf_fig.spectral_analyzer.configuration.overlap_fraction = 0.5
        self._frf_fig.grid(row=0, column=3)
        self._nb.add(self._frf_fig, text='Spectral display')

        # Events:
        self.event_start = EventPublisher()
        self.event_stop = EventPublisher()

    def _cb_start_btn(self):
        self.event_start.raise_event()
        self._frf_fig.start()

    def _cb_stop_btn(self):
        self._frf_fig.stop()
        self.event_stop.raise_event()


if __name__ == "__main__":

    # Define signals:
    my_sensor_list = [{'name': 'time', 'unit': 's', 'chan': -1},
                      {'name': 'OS upper', 'unit': 'mu', 'chan': 0},
                      {'name': 'OS lower', 'unit': 'mu', 'chan': 1},
                      {'name': 'DS upper', 'unit': 'mu', 'chan': 2},
                      {'name': 'DS lower', 'unit': 'mu', 'chan': 3},
                      {'name': 'OS top', 'unit': 'mu', 'chan': 5},
                      {'name': 'OS top center', 'unit': 'mu', 'chan': 6},
                      {'name': 'DS top', 'unit': 'mu', 'chan': 7},
                      {'name': 'DS top center', 'unit': 'mu', 'chan': 8},
                      ]

    # Define log
    pub_rate = 100
    samps_p_pub = 100
    sampling_rate = pub_rate * samps_p_pub
    my_log_info = LogInfo(description='', sampling_rate=sampling_rate,
                          signal_header=[item['name'] for item in my_sensor_list])
    my_log = AutoSaveLog(filename='test_gui.csv', log_info=my_log_info, log_size=100000)
    mygen = msignals.DataGenerator(n_signals=8, pub_rate=pub_rate, n_samples=samps_p_pub)
    mybuffer = msignals.DataBuffer(buffer_size=5000, n_signals=9)

    # Signals:
    my_signals = msignals.SignalDict()
    for i, item in enumerate(my_sensor_list):
        signal = msignals.Signal(name=item['name'], unit=item['unit'], data_index=i, channel_index=item['chan'])
        my_signals[signal.name] = signal

    # Define qtgui
    my_gui = TestGUI(my_signals)


    def start_cb(_):
        mygen.start()
        my_log.reset()


    def stop_cb(_):
        my_log.autosave = False
        mygen.stop()


    # Define qtgui event subscribers
    sub_start = SimpleEventSubscriber(start_cb)
    sub_stop = SimpleEventSubscriber(stop_cb)

    # Define connections:
    mygen.connect(mybuffer)
    mybuffer.connect(my_signals)
    mygen.connect(my_log)

    my_gui.event_start.connect(sub_start)
    my_gui.event_stop.connect(sub_stop)
    my_gui.mainloop()
