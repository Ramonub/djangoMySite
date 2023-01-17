import tkinter as tk

import madernpytools.DAQ as mdaq
import madernpytools.gui as mgui
import madernpytools.frequency_response as mfrf


class GabMeasurementGUI(tk.Frame):
    def __init__(self, task: mdaq.AbstractMadernTask, master=None):
        tk.Frame.__init__(self, master)
        self._task = task

        self.grid()

        self._file_browser = mgui.FileBrowser(self)
        self._file_browser.grid(column=1, row=0, columnspan=2)

        # Signal list
        self._signal_list = mgui.SignalList(self, [item for key, item in sorted(self._task.signals.items(),
                                                                                key=lambda el: el[1].log_index)]
                                            )
        self._signal_list.grid(column=0, row=1, sticky=tk.NW)

        # Spectral plot:
        signal_pairs = []
        for key, item in sorted(self._task.signals.items(), key=lambda el: el[1].log_index):
            if key != 'time':
                signal_pairs.append(self._task.signals.get_signal_pair('time', key))
        #self._fig = mgui.SignalPlot(self, signal_pairs, height=300, width=400, window=20000)
        config = mfrf.SpectralAnalyzerConfiguration(csd_args={'fs': self._task.sampling_rate}, i_input=3, i_output=1)
        self._fig = mgui.SpectralPlot(self, config)
        self._task.event_new_data.connect(self._fig)
        self._fig.grid(column=1, row=1, rowspan=2, columnspan=2)
        #self._fig.ax.set_ylim([-50, 50])
        #self._fig.ax.grid()
        #self._fig.relim = 'x'

        # Buttons:
        self._btn_start = tk.Button(self, text='Start', command=self._cb_start_btn)
        self._btn_start.grid(column=0, row=0, sticky=tk.W)
        self._btn_stop = tk.Button(self, text='Stop', command=self._cb_stop_btn)
        self._btn_stop.grid(column=0, row=0, sticky=tk.E)

        # Handle auto-refresh
        self._auto_refresh = mgui.AutoRefreshWidget(self)
        self._auto_refresh.connect(self._fig)
        self._auto_refresh.connect(self._signal_list)
        self._auto_refresh.refresh_time = 100  # ms

    def _cb_start_btn(self):
        # Try to start DAQ
        self._task.start()
        self._auto_refresh.activate()

    def _cb_stop_btn(self):
        self._task.stop()
        self._auto_refresh.deactivate()


if __name__ == "__main__":

    my_sensor_list = [{'name': 'Acc1', 'unit': '', 'chan': 0},
                      {'name': 'Acc2', 'unit': 'mu', 'chan': 1},
                      {'name': 'Hammer', 'unit': 'mu', 'chan': 2},
                      {'name': 'DS lower', 'unit': 'mu', 'chan': 3},
                      {'name': 'OS top', 'unit': 'mu', 'chan': 5},
                      {'name': 'OS top center', 'unit': 'mu', 'chan': 6},
                      {'name': 'DS top', 'unit': 'mu', 'chan': 7},
                      {'name': 'DS top center', 'unit': 'mu', 'chan': 8},
                      ]

    # Create and reserve device
    gab_task = mdaq.ContinuousMadernTask(task_name='FRF', chassis_name='MadernRD')
    gab_task.sensors.add_pcb_accelerometer(dev_name='MadernRDMod1', ai_index=0)
    gab_task.sensors.add_pcb_accelerometer(dev_name='MadernRDMod1', ai_index=1)
    gab_task.sensors.add_pcb_impulsehammer(dev_name='MadernRDMod1', ai_index=2)

    # setup timing:
    rate = int(10e3)
    buffer_size = int(20e3)
    gab_task.configure(sampling_rate=rate, buffer_size=buffer_size)

    gui = GabMeasurementGUI(task=gab_task)
    gui.mainloop()
    gab_task.close()

