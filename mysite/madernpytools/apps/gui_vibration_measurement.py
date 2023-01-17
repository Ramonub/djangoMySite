import tkinter as tk
import tkinter.ttk as ttk

import madernpytools.data_processing.DAQ as mdaq
import madernpytools.qtgui as mgui
import madernpytools.log as mlog
import madernpytools.backbone as mbb
import madernpytools.signal_handling as msig


class VibrationMeasurementGUI(mgui.SignalDisplay):

    def __init__(self, master=None):
        mgui.SignalDisplay.__init__(self)

        self._frame = ttk.Frame(master=master)
        self._frame.grid()

        # ----- GUI elements ----------
        self.file_browser = mgui.FileBrowser(self._frame)
        self.file_browser.grid(column=1, row=0, columnspan=2)

        # signal visualization:
        self._signal_list = mgui.SignalListWidget(self._frame)
        self._signal_list.grid(column=0, row=1)

        self._nb = ttk.Notebook(self._frame)
        self._nb.grid(column=1, row=1, columnspan=2)

        self._fig = mgui.SignalPlot(self._frame, height=300, width=500)
        self._fig.ax.set_xlabel('time')
        self._fig.ax.set_ylabel(r'Acceleration [$m/s^2$]')
        self._nb.add(self._fig, text='Time-based')

        self._frf_fig = mgui.SpectralPlot(self._frame, height=300, width=500)
        self._frf_fig.spectral_analyzer.configuration.csd_args['fs'] = 2000
        self._frf_fig.spectral_analyzer.configuration.csd_args['nperseg'] = 50
        self._frf_fig.ax.set_xlabel('Frequency [Hz]')
        self._frf_fig.ax.set_ylabel('Acceleration [dB]')
        self._frf_fig.ax.grid()
        self._nb.add(self._frf_fig, text='Spectral-based')

        # Buttons:
        self._btn_start = tk.Button(self._frame, text='Start', command=self._cb_start_btn)
        self._btn_start.grid(column=0, row=0, sticky=tk.NW)
        self._btn_stop = tk.Button(self._frame, text='Stop', command=self._cb_stop_btn)
        self._btn_stop.grid(column=0, row=0, sticky=tk.NE)
        self.save_log = tk.IntVar()
        self._chck_save_log = tk.Checkbutton(self._frame, text='Save log', variable=self.save_log,
                                             command=self._cb_check_log)
        self._chck_save_log.grid(column=0, row=4, sticky=tk.SW)

        # Log Description
        self._description = tk.StringVar()
        self._entry_description = tk.Entry(master=self._frame, textvariable=self._description, width=70)
        self._entry_description.grid(column=1, row=3, sticky=tk.W)
        self._label = tk.Label(self._frame, text='Log description')
        self._label.grid(column=0, row=3)

        # Handle auto-refresh
        self._auto_refresh = mgui.AutoRefreshWidget(self._frame)
        self._auto_refresh.connect(self._fig)
        self._auto_refresh.connect(self._signal_list)
        self._auto_refresh.connect(self._frf_fig)
        self._auto_refresh.refresh_time = 100  # ms

        # Events:
        self.event_start = mbb.EventPublisher()
        self.event_stop = mbb.EventPublisher()
        self.event_log_state_change= mbb.EventPublisher(data_type=int)
        self._nb.bind("<<NotebookTabChanged>>", self._cb_display_tab_changed)

    @property
    def description(self):
        return self._description.get()

    def _cb_check_log(self):
        self.event_log_state_change.raise_event(self.save_log.get())

    def _cb_display_tab_changed(self, event: tk.Event):

        if isinstance(event.widget, ttk.Notebook):
            nb = event.widget
            # Disconnect other tabwidgets:
            for wname in nb.tabs():
                tab_widget = nb.nametowidget(wname)
                if isinstance(tab_widget, mgui.RefreshWidgetSubscriber):
                    self._auto_refresh.disconnect(tab_widget)

            # Connect newly selected tab widget:
            tab_widget = nb.nametowidget(nb.select())
            self._auto_refresh.connect(tab_widget)

    def initialize(self, signals):
        self._fig.initialize(signals)
        self._signal_list.initialize(signals)
        self._frf_fig.initialize(signals)

    def _cb_start_btn(self):
        # Try to start DAQ
        self.event_start.raise_event()
        self._auto_refresh.activate()

    def _cb_stop_btn(self):
        self.event_stop.raise_event()
        self._auto_refresh.deactivate()

    def mainloop(self):
        self._frame.mainloop()

    def cb_signal_update(self, publisher):
        self._fig.cb_signal_update(publisher)
        self._signal_list.cb_signal_update(publisher)
        self._frf_fig.cb_signal_update(publisher)


if __name__ == "__main__":

    # Settings:
    sampling_rate  = int(10e3)
    task_buffer    = int(50e3)
    log_buffer     = int(50e3)
    display_buffer = int(20e3)

    data_step = int(1)

    my_sensor_list = [{'name': 'Time', 'unit': 's', 'chan': -1},
                      {'name': 'Acc1', 'unit': 'm/s2', 'chan': 0},
                      {'name': 'Acc2', 'unit': 'm/s2', 'chan': 1},
                      ]

    # Create DAQ Task:
    gap_task = mdaq.ContinuousMadernTask(task_name='gap measurement', chassis_name='MadernRD')
    for item in my_sensor_list[1:]:
        gap_task.ai_sensors.add_pcb_accelerometer(dev_name='MadernRDMod1', ai_index=item['chan'])
    gap_task.configure(sampling_rate=sampling_rate, buffer_size=task_buffer)
    sampling_rate = gap_task.sampling_rate # Update sample rate to real value

    # Log
    my_log_info = mlog.LogInfo(description='Acceleration measurement', sampling_rate=gap_task.sampling_rate,
                               signal_header=[item['name'] for item in my_sensor_list])
    my_log = mlog.AutoSaveLog(filename='test_gui.csv', log_info=my_log_info, log_size=log_buffer)

    # Signals:
    my_signals = msig.SignalCreator(data_step=data_step, buffer_size=display_buffer)
    for i, item in enumerate(my_sensor_list[1:]):
        my_signals.signal_selector.add_selection([0, i+1], name='{1}'.format('time', item['name']), unit=item['unit'])

    # GUI
    gui = VibrationMeasurementGUI()
    gui.initialize(signals=my_signals.signals)
    gui._frf_fig.spectral_analyzer.configuration.csd_args['fs'] = sampling_rate / data_step
    gui._frf_fig.spectral_analyzer.configuration.csd_args['nperseg'] = 0.1 * display_buffer
    gui._frf_fig.spectral_analyzer.configuration.csd_args['noverlap'] = 0.1 * display_buffer * 0.5

    def log_state_cb(publisher):
        if publisher.get_data()==1:
            my_log.set_filename(gui.file_browser.full_filename)
            my_log_info['description'] = gui.description
            my_log.autosave = True
            print('autosave activated')
        else:
            my_log.autosave = False
            print('autosave deactivated')

    def start_cb(publisher:VibrationMeasurementGUI):
        my_log.set_filename(gui.file_browser.full_filename)
        my_log_info['description'] = gui.description
        my_log.reset()
        my_signals.reset()
        gap_task.start()

    def stop_cb(_):
        gap_task.stop()

    # Define qtgui event subscribers
    sub_start = mbb.SimpleEventSubscriber(start_cb)
    sub_stop = mbb.SimpleEventSubscriber(stop_cb)
    sub_log_check = mbb.SimpleEventSubscriber(log_state_cb)

    # Define connections:
    gap_task.connect(my_signals)
    gap_task.connect(my_log)
    my_signals.connect(gui)
    gui.event_start.connect(sub_start)
    gui.event_stop.connect(sub_stop)
    gui.event_log_state_change.connect(sub_log_check)
    gui.mainloop()

    gap_task.close()
