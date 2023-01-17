import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox as mbox
import xml.etree.cElementTree as ET
import madernpytools.data_processing.DAQ as mdaq
import madernpytools.qtgui as mgui
import madernpytools.log as mlog
import madernpytools.tools.frequency_response as mfrf
import madernpytools.backbone as mbb
import madernpytools.signal_handling as msignals

class GapMeasurementGUI(tk.Frame):

    def __init__(self, master=None, signals=None, xml_settings=None):
        tk.Frame.__init__(self, master)
        self.grid()

        self._signals = signals

        # Log controls:
        self._record_widget = mgui.LogWidget(self)
        self._record_widget.grid(column=1, row=0)

        # Note book for data displays:
        self._nb = ttk.Notebook(self)
        self._nb.grid(column=1, row=1, columnspan=2)

        # ---------------------------------------------------------
        settings = xml_settings.find('.display_settings[@name="sensor_values"]')
        # Sensor value display
        self._sensor_values = mgui.SignalVisualizer(self, signals, settings)
        self._nb.add(self._sensor_values, text='Sensor values')

        # Zero button:
        self._btn_set_zero = tk.Button(self._sensor_values, text='Zero Sensors', command=self._cb_zero)
        self._btn_set_zero.place(relx=0.5, rely=0.10, anchor=tk.CENTER)

        # Initial gap entries:
        self._init_gap = {}
        self._gap_lbl = tk.LabelFrame(self._sensor_values, text='Initial gap values [mu]',
                                      width=340, height=45)
        self._gap_lbl.place(relx=0.48, rely=0.83, anchor=tk.CENTER)
        for loc, relx in [('DS', 0.07), ('MID', 0.5), ('OS', 0.93)]:
            var = tk.DoubleVar()
            wid = tk.Entry(self._gap_lbl, textvariable=var, width=5, )
            wid.place(relx=relx, anchor=tk.CENTER, rely=0.5)
            wid.bind('<FocusOut>', self._cb_leave_init_gap_value)
            self._init_gap[loc] = {'var': var, 'widget': wid}

        # ---------------------------------------------------------
        # Time plot
        self._fig = mgui.TimePlot(self, signal_dict=signals, height=300, width=500)
        self._fig.ax.set_xlabel('time')
        self._fig.ax.set_ylabel(r'')
        self._fig.ax.grid()
        self._nb.add(self._fig, text='Time-based')

        # Spectral plot
        self._frf_fig = mgui.SpectralPlot(self, signal_dict=signals, height=300, width=500)
        self._frf_fig.spectral_analyzer.configuration = mfrf.SpectralAnalyzerConfiguration.from_xml(
            settings.find('./spectral_settings'))
        self._frf_fig.ax.set_xlabel('Frequency [Hz]')
        self._frf_fig.ax.set_ylabel('Displacement [dB, ref. $1\mu m$]')
        self._nb.add(self._frf_fig, text='Spectral-based')

        # ---------------------------------------------------------
        # Events:
        self._sub_record = mbb.SimpleEventSubscriber(self._cb_record)
        self._sub_stop = mbb.SimpleEventSubscriber(self._cb_stop)
        self.event_initial_gap_changed = mbb.EventPublisher(data_type=dict)

        self._nb.bind('<<NotebookTabChanged>>', self._cb_tab_changed)
        self._record_widget.event_record.connect(self._sub_record)
        self._record_widget.event_stop.connect(self._sub_stop)

    def _cb_record(self, _):
        self._btn_set_zero.config(state=tk.DISABLED)

    def _cb_stop(self, _):
        self._btn_set_zero.config(state=tk.NORMAL)

    def _cb_zero(self):
        for key, item in self._signals.items():
            item.set_zero()

    def _cb_tab_changed(self, event):
        widget = event.widget

        # Activate widgets on activated tab:
        for wid in [self._frf_fig, self._fig, self._sensor_values]:
            if str(wid) == widget.select():
                wid.set_auto_refresh(True)
            else:
                wid.set_auto_refresh(False)

    def _cb_leave_init_gap_value(self, e):
        # Event raised:
        self.event_initial_gap_changed.raise_event(self.initial_gap)

    @property
    def initial_gap(self):
        """Get initial gap values"""
        values = {}
        for key, item in self._init_gap.items():
            values[key] = item['var'].get()
        return values

    @property
    def log_controls(self):
        return self._record_widget


class GapComputer(mbb.SimpleEventSubscriber):

    def __init__(self, sensor_scale=10):

        mbb.SimpleEventSubscriber.__init__(self, self.cb_new_data)
        self.sub_initial_gap = mbb.SimpleEventSubscriber(h_callback=self._cb_init_gap)
        self._init_gabs = {'OS': 0.0, 'DS': 0.0}
        self._scale = sensor_scale
        self._gap_data = {}

        self.event_new_data = mbb.EventPublisher(data_type=dict)
        self._signals = {}

    def set_initial_gaps(self, initial_gaps: dict):
        self._init_gabs = initial_gaps

    def set_zero_values(self, zero_values: dict):
        self._zero_values = zero_values

    def _cb_init_gap(self, publisher):
        self._init_gabs = publisher.get_data()

    def _compute_gaps(self):
        """ private method used to compute the gap and bending of cylinders

        :return:
        """

        sides = ['OS', 'DS']
        gaps = {}

        # Compute gap on sides:
        for side in sides:
            up_key = '{0}_{1}'.format(side, 'upper')
            low_key = '{0}_{1}'.format(side, 'lower')

            up_val = self._scale*self._signals[up_key].data
            low_val = self._scale*self._signals[low_key].data

            gaps[side] = self._init_gabs[side] + up_val + low_val

        # Compute bending:
        distance_vals = {}
        #for key in ['OS_top', 'DS_top', 'OS_mid', 'DS_mid']:
        #    distance_vals[key] = self._scale*self._signals[key].data

        #top_avg = 0.5*(distance_vals['OS_top'] + distance_vals['DS_top'])
        #bend_os = distance_vals['OS_mid'] - top_avg
        #bend_ds = distance_vals['DS_mid'] - top_avg

        # Estimate gaps in tool
        gap_avg = 0.5*(gaps['OS'] + gaps['DS'])
        gaps['OS_mid'] = -1 #-2*bend_os + gap_avg
        gaps['DS_mid'] = -1 #-2*bend_ds + gap_avg
        gaps['time_signal'] = self._signals['time'].data

        self._gap_data = gaps

    def cb_new_data(self, publisher):
        # Update signals:
        self._signals = publisher.get_data()

        # Compute gap_data:
        self._compute_gaps()

        # Raise update event:
        self.event_new_data.raise_event(self._gap_data)


class GapPlot(tk.Frame):

    def __init__(self, parent=None):
        tk.Frame.__init__(self, parent)

        # Define subscriber to data
        self.subcriber_gap = mbb.SimpleEventSubscriber(h_callback=self._new_gap_data)

        # Define matplotlib figure
        self.gap_dict = {}

    def _new_gap_data(self, publisher):
        self.gap_dict = publisher.get_data()

        # Update lines:


def load_sensors(gap_task: mdaq.MadernTask, settings_tree, devices):
    """
    """
    for sensor_setting in settings_tree.find('sensor_settings').iter('sensor'):
        # Check if device is available:
        if sensor_setting.find('device_name').text in devices.device_names:
            if sensor_setting.find('type').text == 'capacitive':
                gap_task.add_sensor(mdaq.MEcapaNCDTSensor(name=sensor_setting.find('name').text,
                                                          device_name=sensor_setting.find('device_name').text,
                                                          channel_index=int(sensor_setting.find('channel').text)
                                                          )
                                    )
            elif sensor_setting.find('type').text == 'acceleration':
                gap_task.add_sensor(mdaq.PCBAccelerationSensor(name=sensor_setting.find('name').text,
                                                               device_name=sensor_setting.find('device_name').text,
                                                               channel_index=int(sensor_setting.find('channel').text)
                                                               )
                                    )

            elif sensor_setting.find('type').text == 'acceleration_bruel':
                gap_task.add_sensor(mdaq.BruelAccelerationSensor(name=sensor_setting.find('name').text,
                                                                 device_name=sensor_setting.find('device_name').text,
                                                                 channel_index=int(sensor_setting.find('channel').text)
                                                                 )
                                    )
            elif sensor_setting.find('type').text == 'impulse_hammer':
                gap_task.add_sensor(mdaq.PCBImpactHammerSensor(name=sensor_setting.find('name').text,
                                                               device_name=sensor_setting.find('device_name').text,
                                                               channel_index=int(sensor_setting.find('channel').text)
                                                               )
                                    )
            elif sensor_setting.find('type').text=='angular_encoder':
                gap_task.add_sensor(mdaq.AngularX4Encoder(name=sensor_setting.find('name').text,
                                                          device_name=sensor_setting.find('device_name').text,
                                                          counter_index=int(sensor_setting.find('channel').text),
                                                          pulses_per_rev=int(sensor_setting.find('pulses_per_rev').text),
                                                          )
                                    )
            elif sensor_setting.find('type').text=='voltage':
                gap_task.add_sensor(mdaq.VoltageSensor(device_name=sensor_setting.find('device_name').text,
                                                       channel_index=int(sensor_setting.find('channel').text),
                                                       name=sensor_setting.find('name').text
                                                       )
                                    )
            elif sensor_setting.find('type').text=='thermocouple':
                gap_task.add_sensor(mdaq.Thermocouple(device_name=sensor_setting.find('device_name').text,
                                                      channel_index=int(sensor_setting.find('channel').text),
                                                      name=sensor_setting.find('name').text
                                                      )
                                    )

        else:
            print(devices.device_names)
            mbox.showerror(title="Device not found", message="Could not find device name {0} for sensor {1}".format(
                sensor_setting.find('device_name').text, sensor_setting.find('name').text))


if __name__ == "__main__":

    # Load Settings from XML file:
    fn = './gui_config/gap_gui_settings_sim.xml'
    settings_tree = ET.parse(fn).getroot()
    ds = settings_tree.find('daq_settings')
    chassis_name = ds.find('chassis_name').text
    task_name = ds.find('task_name').text
    sampling_rate = int(ds.find('sampling_rate').text)
    buffer_size = int(ds.find('buffer_size').text)
    log_size = int(ds.find('log_size').text)

    # -------------- FOR REAL MEASUREMENT ----------------------------------
    devices = mdaq.list_devices()

    chassis_found = False
    try:
        print('Trying to connect to NI-cDAQ module named \'{0}\'...'.format(chassis_name))
        # Try to reset device:
        devices[chassis_name].self_test_device()
        chassis_found = True
    except:
        mbox.showerror(title="Device not found", message="Could not find chassis named {0}".format(chassis_name), parent=None)

    # Create DAQ Task:
    if chassis_found:
        print('Setting up ni_task...')
        gap_task = mdaq.ContinuousMadernTask(task_name=task_name, chassis_name=chassis_name)

        print('Loading sensors...')
        load_sensors(gap_task, settings_tree, mdaq.list_devices())

        # setup timing:
        gap_task.configure(sampling_rate=sampling_rate, buffer_size=buffer_size)
        sampling_rate = gap_task.sampling_rate # Get sampling rate assigned by NI-DAQ (it chooses closest to desired)
        task_signals = gap_task.signals

        # ------- LOG ------------------
        my_log_info = mlog.LogInfo(description='', sampling_rate=sampling_rate,
                                   signal_header=list(gap_task.signals.name.keys()))
        my_log = mlog.AutoSaveLog(filename='test_gui.csv', log_info=my_log_info, log_size=log_size)

        # ------------- DISPLAY SIGNALS ---------------------------

        # Create signals for display
        display_signals = msignals.SignalDict()
        for key, sig in task_signals.items():
            display_signals[key] = msignals.Signal(name=sig.name,
                                                   unit=sig.unit,
                                                   data_index=sig.data_index,
                                                   module=sig.module,
                                                   channel_index=sig.channel_index,
                                                   scale = sig.scale)

        os_gap_sig = msignals.SummedSignal(name='OS_gap',
                                     unit='micron',
                                     data_index_1=display_signals['OS_upper'].data_index,
                                     data_index_2=display_signals['OS_lower'].data_index,
                                     module='None',
                                     channel_index=-1, scale=100)

        ds_gap_sig = msignals.SummedSignal(name='DS_gap',
                                           unit='micron',
                                           data_index_1=display_signals['DS_upper'].data_index,
                                           data_index_2=display_signals['DS_lower'].data_index,
                                           module='None',
                                           channel_index=-1, scale=100)

        display_signals['OS_gap'] = os_gap_sig
        display_signals['DS_gap'] = ds_gap_sig

        data_buffer = msignals.DataBuffer(buffer_size, len(display_signals)-2) # Buffer is used to display longer time period

        # ------------ GUI
        print('Loading GUI...')
        gui = GapMeasurementGUI(signals=display_signals, xml_settings=settings_tree)
        gui.winfo_toplevel().title('Gap display')

        # ---------- EVENT HANDLING ------------------
        def record_cb(publisher: mbb.EventPublisher):
            """"Start log recording"""
            fn = publisher.get_data()
            # Reset the log
            gap_task.reset()
            data_buffer.reset()
            my_log.reset()
            my_log.info['description'] = gui.log_controls.description
            my_log.set_filename(fn)

            # Update calibration values:
            for key, item in display_signals.calibration_value.items():
                my_log_info['calib_{0}'.format(key)] = item

            # Update scale values:
            for key, item in display_signals.scale.items():
                my_log_info['scale_{0}'.format(key)] = item

            # Update Initial gap settings:
            for key, item in gui.initial_gap.items():
                my_log_info['init_gap_{0}'.format(key)] = item

            my_log.autosave = True

        def stop_cb(_):
            """"Stop log recording"""
            my_log.autosave = False

        # Connect events:
        sub_record = mbb.SimpleEventSubscriber(record_cb)
        sub_stop = mbb.SimpleEventSubscriber(stop_cb)

        mygap_computer = GapComputer(sensor_scale=100.0)
        task_signals.connect(mygap_computer)

        # Define connections:
        gap_task.connect(my_log)        # Connect ni_task to log, to store measured signals in log
        gap_task.connect(data_buffer)   # Connect ni_task to data_buffer to create a larger time window for display
        data_buffer.connect(display_signals)  # Connect data_buffer to display_signals, for visualization in GUI
        gui.event_initial_gap_changed.connect(mygap_computer.sub_initial_gap)

        # Connect controls of GUI
        gui.log_controls.event_record.connect(sub_record)  # Record button
        gui.log_controls.event_stop.connect(sub_stop)      # Stop recording button

        # ------------------ MAIN LOOP --------------------
        # Start ni_task and main loop:
        with gap_task as gt:
            gt.start()
            gui.mainloop()
