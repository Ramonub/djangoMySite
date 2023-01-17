import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox as mbox
import xml.etree.cElementTree as ET
import madernpytools.data_processing.DAQ as mdaq
import madernpytools.qtgui as mgui
import madernpytools.log as mlog
import madernpytools.backbone as mbb
import madernpytools.signal_handling as msignals


class DataGUI(tk.Frame):

    def __init__(self, master=None, signals=None, display_settings=None):
        tk.Frame.__init__(self, master)
        self.grid()

        self._signals = signals

        # Log controls:
        self._record_widget = mgui.LogWidget(self)
        self._record_widget.grid(column=1, row=0)

        # Note book for data displays:
        self._nb = ttk.Notebook(self)
        self._nb.grid(column=1, row=1, columnspan=2)

        # Sensors
        self._sensor_values = mgui.SignalVisualizer(self, signals, display_settings)
        self._nb.add(self._sensor_values, text='Sensor values')

        # Time plot
        self._fig = mgui.TimePlot(self, signal_dict=signals, height=300, width=500)
        self._fig.ax.set_xlabel('time')
        self._fig.ax.set_ylabel(r'displacement [$\mu m $]')
        self._fig.ax.grid()
        self._nb.add(self._fig, text='Time-based')

        # Spectral plot
        self._frf_fig = mgui.SpectralPlot(self, signal_dict=signals, height=300, width=500,
                                          spectral_settings=display_settings.find('spectral_settings'))
        self._frf_fig.ax.set_xlabel('Frequency [Hz]')
        self._frf_fig.ax.set_ylabel('Displacement [dB]')
        self._nb.add(self._frf_fig, text='Spectral-based')

        self._btn_set_zero = tk.Button(self._sensor_values, text='Zero Sensors', command=self._cb_zero)
        self._btn_set_zero.place(relx=0.5, rely=0.10, anchor=tk.CENTER)

        # Events:
        self._sub_record = mbb.SimpleEventSubscriber(self._cb_record)
        self._sub_stop = mbb.SimpleEventSubscriber(self._cb_stop)

        self._nb.bind('<<NotebookTabChanged>>', self._cb_tab_changed)
        self._record_widget.event_record.connect(self._sub_record)
        self._record_widget.event_stop.connect(self._sub_stop)

    @property
    def zero_button(self):
        return self._btn_set_zero

    def _cb_zero(self):
        for _, item in self._signals.items():
            item.set_zero()

    def _cb_record(self, _):
        self._btn_set_zero.config(state=tk.DISABLED)

    def _cb_stop(self, _):
        self._btn_set_zero.config(state=tk.NORMAL)

    def _cb_tab_changed(self, event):
        widget = event.widget

        # Activate widgets on activated tab:
        for wid in [self._frf_fig, self._fig, self._sensor_values]:
            if str(wid) == widget.select():
                wid.set_auto_refresh(True)
            else:
                wid.set_auto_refresh(False)

    @property
    def log_controls(self):
        return self._record_widget

    @property
    def initial_gap(self):
        return self._sensor_values.initial_gap


def load_sensors(gap_task: mdaq.MadernTask, settings_tree, devices):
    """
    """
    for sensor_setting in settings_tree.find('sensor_settings').iter('sensor'):
        # Check if device is available:
        if sensor_setting.find('device_name').text in devices.device_names:
            if sensor_setting.find('type').text == 'capacitive':
                gap_task.add_sensor(mdaq.MEcapaNCDTSensor(name=sensor_setting.find('task_name').text,
                                                          device_name=sensor_setting.find('device_name').text,
                                                          channel_index=int(sensor_setting.find('channel').text)
                                                            )
                                    )
            elif sensor_setting.find('type').text == 'acceleration':
                gap_task.add_sensor(mdaq.PCBAccelerationSensor(name=sensor_setting.find('task_name').text,
                                                               device_name=sensor_setting.find('device_name').text,
                                                               channel_index=int(sensor_setting.find('channel').text)
                                                               )
                                    )

            elif sensor_setting.find('type').text == 'acceleration_bruel':
                gap_task.add_sensor(mdaq.BruelAccelerationSensor(name=sensor_setting.find('task_name').text,
                                                                 device_name=sensor_setting.find('device_name').text,
                                                                 channel_index=int(sensor_setting.find('channel').text)
                                                          )
                                    )
            elif sensor_setting.find('type').text == 'impulse_hammer':
                gap_task.add_sensor(mdaq.PCBImpactHammerSensor(name=sensor_setting.find('task_name').text,
                                                               device_name=sensor_setting.find('device_name').text,
                                                               channel_index=int(sensor_setting.find('channel').text)
                                                               )
                                    )
            elif sensor_setting.find('type').text=='angular_encoder':
                gap_task.add_sensor(mdaq.AngularX4Encoder(name=sensor_setting.find('task_name').text,
                                                          device_name=sensor_setting.find('device_name').text,
                                                          counter_index=int(sensor_setting.find('channel').text),
                                                          pulses_per_rev=int(sensor_setting.find('pulses_per_rev').text),
                                                          )
                                    )
            elif sensor_setting.find('type').text=='voltage':
                gap_task.add_sensor(mdaq.VoltageSensor(device_name=sensor_setting.find('device_name').text,
                                                       channel_index=int(sensor_setting.find('channel').text),
                                                       name=sensor_setting.find('task_name').text
                                                       )
                                    )
            elif sensor_setting.find('type').text=='thermocouple':
                gap_task.add_sensor(mdaq.Thermocouple(device_name=sensor_setting.find('device_name').text,
                                                      channel_index=int(sensor_setting.find('channel').text),
                                                      name=sensor_setting.find('task_name').text
                                                     )
                                    )

        else:
            print(devices.device_names)
            mbox.showerror(title="Device not found", message="Could not find device task_name {0} for sensor {1}".format(
                sensor_setting.find('device_name').text, sensor_setting.find('task_name').text))


if __name__ == "__main__":

    # Load Settings from XML file:
    fn = './gui_config/data_gui_sim.xml'
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

        gap_task = mdaq.TaskLoader.load('gap_task.xml')
        #gap_task = mdaq.ContinuousMadernTask(task_name=task_name, chassis_name=chassis_name)

        #print('Loading _sensors...')
        #load_sensors(gap_task, settings_tree, mdaq.list_devices())

        #print('AI _sensors: ', len(gap_task.ai_sensors))
        #print('CI _sensors: ', len(gap_task.ci_sensors))

        # setup timing:
        #gap_task.configure(sampling_rate=sampling_rate, buffer_size=buffer_size,)
        #sampling_rate = gap_task.sampling_rate # Get sampling rate assigned by NI-DAQ (it chooses closest to desired)
        task_signals = gap_task.signals

        #mdaq.TaskLoader.save(gap_task, 'gap_task.xml')

        # ------- LOG ------------------
        my_log_info = mlog.LogInfo(description='', sampling_rate=sampling_rate,
                                   signal_header=gap_task.signals.signal_names)
        my_log = mlog.AutoSaveLog(filename='test_gui.csv', log_info=my_log_info, log_size=log_size)

        # ------------- DISPLAY SIGNALS ---------------------------

        # Create signals for display
        display_signals = msignals.SignalDict()
        for key, sig in task_signals.items():
            display_signals[key] = msignals.Signal(name=sig.task_name,
                                                   unit=sig.unit,
                                                   data_index=sig.data_index,
                                                   module=sig.module,
                                                   channel_index=sig.channel_index,
                                                   scale=sig.scale)

        # Buffer only applies to
        data_buffer = msignals.DataBuffer(buffer_size, len(display_signals)) # Buffer is used to display longer time period

        # Add gap signal
        for side in ['OS', 'DS']:
            if '{0}_lower'.format(side) in display_signals.keys() and '{0}_upper'.format(side) in display_signals.keys():
                display_signals['{}_gap'.format(side)] = msignals.SummedSignal(signal1=display_signals['{}_lower'.format(side)],
                                                                  signal2=display_signals['{}_upper'.format(side)],
                                                                  name='{}_gap'.format(side), unit='mm')



        # ------------ GUI
        print('Loading GUI...')
        gui = DataGUI(signals=display_signals, display_settings=settings_tree.find('display_settings'))
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

            my_log.autosave = True

        def stop_cb(_):
            """"Stop log recording"""
            my_log.autosave = False

        # Connect events:
        sub_record = mbb.SimpleEventSubscriber(record_cb)
        sub_stop = mbb.SimpleEventSubscriber(stop_cb)

        # Define connections:
        gap_task.connect(my_log)        # Connect ni_task to log, to store measured signals in log
        gap_task.connect(data_buffer)   # Connect ni_task to data_buffer to create a larger time window for display
        data_buffer.connect(display_signals)  # Connect data_buffer to display_signals, for visualization in GUI

        # Connect controls of GUI
        gui.log_controls.event_record.connect(sub_record)  # Record button
        gui.log_controls.event_stop.connect(sub_stop)      # Stop recording button

        # ------------------ MAIN LOOP --------------------
        # Start ni_task and main loop:
        with gap_task as gt:
            gt.start()
            gui.mainloop()

