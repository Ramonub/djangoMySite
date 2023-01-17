from enum import Enum
from PIL import Image, ImageTk
import time, datetime, os
import numpy as np
import threading as thrd
import tkinter.ttk as ttk
import tkinter as tk
import tkinter.filedialog
import xml.etree.cElementTree as ET
import nidaqmx
import matplotlib
import matplotlib.cm as cm

import madernpytools.qtgui as mgui
import madernpytools.backbone as mbb
import madernpytools.tools.experiments as mexp
import madernpytools.log as mlog
import madernpytools.plot as mplt
import madernpytools.tools.frequency_response as mfrf

cols = cm.Set1(np.linspace(0, 1, 10))[:, :3]

matplotlib.use('TkAgg')

class AutoResizeFrame(object):

    def __init__(self, frame, cb_resize=None):
        self._frame = frame

        if cb_resize is not None:
            self._frame.bind('<Configure>', cb_resize)
        else:
            self._frame.bind('<Configure>', self._handle_resize)

    def _handle_resize(self, event):
        pass

    @property
    def frame_parent(self):
        return self._frame.nametowidget(self._frame.winfo_parent())


class AbstractNavigationFrame(tk.Frame):

    def __init__(self, master=None, *args):
        tk.Frame.__init__(self, master=master, height=30, width=600)
        self._auto_resize = AutoResizeFrame(self, self._handle_resize)

    def _handle_resize(self, event):
        pass


class AbstractMessageFrame(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master=master, height=50, width=600, bd=3, relief=tk.SUNKEN)
        self._auto_resize = AutoResizeFrame(self, self._handle_resize)

    def _handle_resize(self, event):
        pass


class AbstractVisualizationFrame(tk.Frame):

    def __init__(self, master=None, *args):
        tk.Frame.__init__(self, master=master, height=400, width=600, bd=3)
        self._auto_resize = AutoResizeFrame(self, self._handle_resize)

    def _handle_resize(self, event):
        pass


class BodyFrame(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master=master)
        self.grid()

        # Initialize
        self._message_frame = None
        self.message_frame = AbstractMessageFrame(self)

        self._visualization_frame = None
        self.visualization_frame = AbstractVisualizationFrame(self)

        self._navigation_frame = None
        self.navigation_frame = AbstractNavigationFrame(self)

    @property
    def parent(self):
        return self.nametowidget(self.winfo_parent())

    def _set_frame_attr(self, value, frame_attr, **kwargs):
        old_frame = getattr(self, frame_attr)

        if old_frame is not None:
            old_frame.grid_forget()

        self.__setattr__(frame_attr,  value)
        value.grid(**kwargs)

    @property
    def message_frame(self):
        return self._message_frame

    @message_frame.setter
    def message_frame(self, value):
        self._set_frame_attr(value, '_message_frame', column=0, row=0)

    @property
    def navigation_frame(self):
        return self._navigation_frame

    @navigation_frame.setter
    def navigation_frame(self, value):
        self._set_frame_attr(value, '_navigation_frame', column=0, row=1)

    @property
    def visualization_frame(self):
        return self._visualization_frame

    @visualization_frame.setter
    def visualization_frame(self, value):
        self._set_frame_attr(value, '_visualization_frame', column=0, row=2)

    def _handle_resize(self, event):
        raise NotImplementedError()


class RibbonItem(tk.Button):

    def __init__(self, master, icon_fn='', **kwargs):

        if icon_fn != '':
            self._icon = RibbonItem.load_image(icon_fn)
            tk.Button.__init__(self, master=master, **kwargs, image=self._icon, command=self._cb_click)
        else:
            tk.Button.__init__(self, master=master, **kwargs, command=self._cb_click)

        self.event_click = mbb.EventPublisher()

    def _cb_click(self):
        self.event_click.raise_event()

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as im_handle:
            im = Image.open(im_handle)
            photo = ImageTk.PhotoImage(im)
            im.close()
        return photo


class RibbonFrame(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master, height=50, width=600)
        self.pack_propagate(0)

        self.btn_new = RibbonItem(self, icon_fn='./figures/icon_new_modal_test.png')
        self.btn_new.pack(side=tk.LEFT)

        self.btn_open = RibbonItem(self, icon_fn='./figures/icon_open_modal_test.png')
        self.btn_open.pack(side=tk.LEFT)

    def set_state(self, state=tk.NORMAL):
        self.btn_new.configure(state=state)
        self.btn_open.configure(state=state)


class SimpleMessageFrame(AbstractMessageFrame):

    def __init__(self, master=None, message='', title=''):
        AbstractMessageFrame.__init__(self, master)
        self.grid_propagate(0)
        self.grid(sticky=tk.NW)
        self.message_lbl = tk.Label(self, text=message)
        self.message_lbl.grid(padx=10, column=0, row=1, sticky=tk.NW)

        self.title_lbl = tk.Label(self, text=title, font=('Times', 10, 'bold'))
        self.title_lbl.grid(padx=10, column=0, row=0, sticky=tk.NW)

    @property
    def text(self):
        return self.message_lbl.cget('text')

    @text.setter
    def text(self, value):
        self.message_lbl.config(text=value)

    @property
    def title(self):
        return self.title_lbl.cget('text')

    @title.setter
    def title(self, value):
        self.title_lbl.config(text=value)


class BackContinueFrame(AbstractNavigationFrame):

    def __init__(self, master=None):
        AbstractNavigationFrame.__init__(self, master)
        self.pack_propagate(0)

        self.btn_continue = tk.Button(self, text='Continue', command=self._cb_continue)
        self.btn_continue.pack(side=tk.RIGHT, ipadx=5)
        self.btn_back = tk.Button(self, text='Back', command=self._cb_back)
        self.btn_back.pack(side=tk.RIGHT)

        self.event_continue_click = mbb.EventPublisher()
        self.event_back_click = mbb.EventPublisher()

    def _cb_back(self):
        self.event_back_click.raise_event()

    def _cb_continue(self):
        self.event_continue_click.raise_event()


class TestResultDisplay(AbstractVisualizationFrame):

    def __init__(self, master=None, spectral_settings=None):
        AbstractVisualizationFrame.__init__(self, master=master)

        default_sets = mfrf.SpectralAnalyzerConfiguration(i_input=3, i_output=1,
                                                          filter_window=('exponential', None, 1e6))

        self._spectral_settings = mbb.ArgumentVerifier(mfrf.SpectralAnalyzerConfiguration,
                                                       default_sets).verify(spectral_settings)
        self._spectral_plot = mplt.SpectralPlot()

        # Notebook for data displays:
        self._nb = ttk.Notebook(master=self)
        self._nb.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Create plots
        self._figs = {}
        for key in ['time', 'spectral']:
            gui_plt = mgui.GUIPlot(self, width=500, height=300)
            self._nb.add(gui_plt, text=key)
            self._figs[key] = {'figure': gui_plt.fig, 'container': gui_plt}

        # Axes:
        self._figs['time']['ax1'] = self._figs['time']['figure'].add_subplot(111)
        self._figs['time']['ax2'] = self._figs['time']['ax1'].twinx()

        self._figs['spectral']['magnitude'] = self._figs['spectral']['figure'].add_subplot(111)
        self._figs['spectral']['coherence'] = self._figs['spectral']['magnitude'].twinx()

    def load_log(self, log):

        data = log.data

        # Spectral analyzer:
        self._figs['spectral']['magnitude'].clear()
        self._figs['spectral']['coherence'].clear()

        self._spectral_settings.sampling_rate = log.info.sampling_rate
        analyzer = mfrf.SpectralAnalyzer(configuration=self._spectral_settings)
        analysis_result = analyzer.analyze(data)
        self._spectral_plot = mplt.SpectralPlot(analysis_result, axs=self._figs['spectral'], plot_keys=['Sff', 'H'],
                          plot_coherence=False)
        self._spectral_plot.set_colordict({'Sff': 'orange', 'H': 'purple'})

        self._figs['spectral']['magnitude'].grid()
        self._figs['spectral']['container'].refresh()

        # Time plot:
        ax = self._figs['time']['ax2']
        ax.clear()
        l_in, = ax.plot(data[:, 0], data[:, self._spectral_settings.i_input], label='Input', color='orange')
        ax.set_xlabel(r'Time [s]')
        ax.set_ylabel(r'Impact [N]')

        ax2 = self._figs['time']['ax1']
        ax2.clear()
        l_out, = ax2.plot(data[:, 0], data[:, self._spectral_settings.i_output], color=cols[1,], label='Response')
        ax2.set_ylabel(r'Response [$m/s^2$]')
        ax2.grid()
        ax2.autoscale_view()
        ax.legend([l_in, l_out], ['Impact', 'Response 0'])
        self._figs['time']['container'].refresh()


class ToolSetVisualization(AbstractVisualizationFrame):

    def __init__(self, master=None):
        AbstractVisualizationFrame.__init__(self, master=master)

        self._widgets = {}

        # Load images:
        self._photo_tooling = None
        self._tool_label = tk.Label(self, width=600, height=400)
        self._tool_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Tool Ids & Selection:
        self.tools_id_visible = True
        self._upper_id = mgui.LabeledEntry(self, label='Tool ID ')
        self._upper_id.place(relx=0, rely=0)
        self._upper_id.config(width=8)
        self._upper_check_var = tk.IntVar()
        self._upper_check_var.set(1)
        self._upper_check = tk.Checkbutton(self, variable=self._upper_check_var)
        self._widgets['upper_id'] = self._upper_id
        self._widgets['upper_check'] = self._upper_check

        self._lower_id = mgui.LabeledEntry(self, label='Tool ID ')
        self._lower_id.config(width=8)
        self._lower_id.place(relx=0, rely=0)
        self._lower_check_var = tk.IntVar()
        self._lower_check_var.set(1)
        self._lower_check = tk.Checkbutton(self, variable=self._lower_check_var)
        self._widgets['lower_id'] = self._lower_id
        self._widgets['lower_id_check'] = self._lower_check

        self.clear_canvas()

    def clear_canvas(self):
        for w, item in self._widgets.items():
            item.place_forget()

        self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_tooling.png')
        self._tool_label.config(image=self._photo_tooling, width=self.cget('width'), height=self.cget('height'))

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as im_handle:
            im = Image.open(im_handle)
            photo = ImageTk.PhotoImage(im)
            im.close()
        return photo

    def show_tool_id(self):
        self._upper_id.place(relx=0.45, rely=0.38, anchor=tk.CENTER)
        self._upper_check.place(relx=0.39, rely=0.38, anchor=tk.CENTER)

        self._lower_check.place(relx=0.39, rely=0.63, anchor=tk.CENTER)
        self._lower_id.place(relx=0.45, rely=0.63, anchor=tk.CENTER)

    def show_sensor(self, location=''):
        if location == 'upper':
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_sensor_upper.png')
        elif location == 'lower':
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_sensor_lower.png')
        else:
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_tooling.png')
        self._tool_label.config(image=self._photo_tooling, width=self.cget('width'), height=self.cget('height'))

    def show_impact(self, location=''):
        if location == 'upper':
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_impact_upper.png')
        elif location == 'lower':
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_impact_lower.png')
        else:
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_tooling.png')
        self._tool_label.config(image=self._photo_tooling, width=self.cget('width'), height=self.cget('height'))

    def show_impacted(self, location):
        if location == 'upper':
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_impacted_upper.png')
        elif location == 'lower':
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_impacted_lower.png')
        else:
            self._photo_tooling = ToolSetVisualization.load_image('../figures/ema_schematic_tooling.png')
        self._tool_label.config(image=self._photo_tooling, width=self.cget('width'), height=self.cget('height'))

    @property
    def tool_info(self):
        self.clear_canvas()

        return {'upper': (self._upper_id.value, self._upper_check_var.get()),
                'lower': (self._lower_id.value, self._lower_check_var.get())}

    @property
    def path(self):
        raise NotImplementedError()

    @path.setter
    def path(self, value):
        raise NotImplementedError()


class GuiStates(Enum):
    IDLE = 0
    INITIALIZING = 1
    WAITING_FOR_IMPACT = 2


class EMAGui(tk.Frame):

    def __init__(self, master=None, spectral_settings=None):
        tk.Frame.__init__(self, master=master)
        self.grid()
        self._ribbon = RibbonFrame(master)
        self._ribbon.grid(column=0, row=0)

        self._body = BodyFrame(master)
        self._body.grid(column=0, row=1)

        self._toolset_visualizer = ToolSetVisualization(self._body)
        self._toolset_visualizer.grid_forget()
        self._data_visualizer = TestResultDisplay(self._body, spectral_settings=spectral_settings)

        self._navigation = BackContinueFrame(self._body)           # Navigation buttons

        self._body.message_frame = SimpleMessageFrame(self._body, message='')    # Message display
        self._body.navigation_frame = AbstractNavigationFrame(self._body)
        self._body.visualization_frame = AbstractVisualizationFrame(self._body)  # Visualization area

        self._subs = []
        self._subs.append(mbb.SimpleEventSubscriber(h_callback=self._cb_start))
        self.event_start.connect(self._subs[-1])

        self._subs.append(mbb.SimpleEventSubscriber(h_callback=self._cb_back))
        self._navigation.event_back_click.connect(self._subs[-1])

    def set_state(self, state: GuiStates):

        if state == GuiStates.INITIALIZING:
            self.show_initializing()

        elif state == GuiStates.IDLE:
            self.show_idle()

    def _cb_start(self, _):
        self._body.navigation_frame = self._navigation
        self._body.navigation_frame.btn_back.config(state=tk.DISABLED)
        self._ribbon.set_state(tk.DISABLED)

    def _cb_back(self, _):
        self._navigation.btn_back.config(state=tk.DISABLED)

    @property
    def ribbon(self):
        return self._ribbon

    @property
    def event_continue(self):
        """ Event raised when continue button is pressed
        """
        return self._navigation.event_continue_click

    @property
    def event_back(self):
        """Event raised when back button is pressed."""
        return self._navigation.event_back_click

    @property
    def event_start(self):
        """Button pressed when start button is pressed."""
        return self._ribbon.btn_new.event_click

    @property
    def event_open(self):
        """Button pressed when start button is pressed."""
        return self._ribbon.btn_new.event_click

    def set_message(self, message):
        self._body.message_frame.text = message

    def set_title(self, title):
        self._body.message_frame.title = title

    def show_toolset_info_form(self):
        self._body.message_frame.text = "Please insert cylinder_properties ids in the fields below."
        self._body.visualization_frame = self._toolset_visualizer
        self._toolset_visualizer.clear_canvas()
        self._toolset_visualizer.show_tool_id()

    def show_sensor_placement(self, location=''):
        self._body.visualization_frame = self._toolset_visualizer
        self._toolset_visualizer.clear_canvas()
        self._toolset_visualizer.show_sensor(location)

        self._body.message_frame.text = "Place acceleration sensor on indicated location and press \'continue\'."

    def show_impact_request(self, location=''):
        self._body.message_frame.text = 'Impact tool on indicated location to start measurement.'

        self._body.visualization_frame = self._toolset_visualizer
        self._toolset_visualizer.clear_canvas()
        self._toolset_visualizer.show_impact(location)

    def show_impact_detected(self, location=''):
        self._body.visualization_frame = self._toolset_visualizer
        self._body.message_frame.text = 'Impact has been detected.'
        self._toolset_visualizer.show_impacted(location)

    def show_file_saved(self):
        self._body.message_frame.text = 'File saved'

    def show_initializing(self):
        self._body.message_frame.text = 'Initializing data acquisition device'
        self._body.visualization_frame = self._toolset_visualizer
        self._body.navigation_frame = AbstractNavigationFrame(self)
        self._toolset_visualizer.clear_canvas()
        self._ribbon.set_state(tk.DISABLED)

    def show_impact_timeout(self):
        self._body.message_frame.text = 'Impact timed out.'

    def show_idle(self):
        self._ribbon.set_state(tk.NORMAL)
        self._body.message_frame.text = 'Idle'

    def show_test_result(self, log):
        self._body.message_frame.text = 'Test results are displayed below. Press \'Continue\' to proceed, or \'Back\' to redo the measurement.'
        self._navigation.btn_back.config(state=tk.NORMAL)

        self._data_visualizer.load_log(log)
        self._body.visualization_frame = self._data_visualizer

    def show_finished(self):
        self._body.message_frame.text = 'The test has finished.'

        self._toolset_visualizer.clear_canvas()
        self._body.visualization_frame = self._data_visualizer

        self._body.navigation_frame = AbstractNavigationFrame(self._body)
        self._ribbon.set_state(tk.NORMAL)

    def get_tool_info(self):
        return self._body.visualization_frame.tool_info


class LoadTask(mbb.BackgroundJob):

    def __init__(self, configuration, simulate=True):
        mbb.BackgroundJob.__init__(self)
        self._conf = configuration
        self._modal_test = None
        self._sim = simulate

    def _work_sequence(self):
        if not self._sim:
            try:
                self._modal_test = mexp.ModalTest(self._conf)
            except nidaqmx.DaqError:
                raise
                self._modal_test = mexp.SimulateModalTest(self._conf)
                self.abort()
        else:
            self._modal_test = mexp.SimulateModalTest(self._conf)

    def get_result(self):
        return mbb.JobResult(job_state=self.state, data=self._modal_test)


class EMAWrapper(object):

    def __init__(self, xml_filename=None, simulate=False):

        # Flags:
        self._continue_clicked = False
        self._back_clicked = False
        self._current_location = ''
        self._simulate = simulate

        self._xml_filename = mbb.ArgumentVerifier(str, './gui_config/settings_ema_gui.xml').verify(xml_filename)
        print(self._xml_filename)

        # Load XML
        if os.path.exists(self._xml_filename):
            xml_root = ET.parse(self._xml_filename).getroot()
            self._xml_conf = xml_root
        else:
            xml_root = EMAWrapper.create_xml_settings()
            self._xml_conf = xml_root

            # Store new xml settings:
            tree = ET.ElementTree(self._xml_conf)
            tree.write(self._xml_filename)

        # Extract Wrapper settings:
        self._n_reps = eval(xml_root.find('n_reps').text)
        self._dir = xml_root.find('dir').text

        # Setup GUI:
        frf_sets = mfrf.SpectralAnalyzerConfiguration.from_xml(xml_root.find('spectral_settings'))
        self.gui = EMAGui(spectral_settings=frf_sets)
        frame_name = 'SIMULATED Modal Test' if simulate else 'Modal Test'
        self.gui.winfo_toplevel().title(frame_name)
        self.gui.set_state(GuiStates.INITIALIZING)

        # Setup ni_task:
        print(xml_root)
        task_configuration = mexp.ModalTestConfiguration.from_xml(xml_root)
        self._modal_test = mexp.AbstractModalTest()
        self._task_finished = mbb.SimpleEventSubscriber(h_callback=self._cb_task_loaded)
        self._task_loader = mbb.BackgroundWorker(LoadTask(configuration=task_configuration, simulate=simulate))
        self._task_loader.connect(self._task_finished)
        self._task_loader.start()

        # event subscribers:
        self._sub_impact_request = mbb.SimpleEventSubscriber(self._cb_waiting_for_impact)
        self._sub_impact_detected = mbb.SimpleEventSubscriber(self._cb_impact_detected)
        self._sub_file_saved = mbb.SimpleEventSubscriber(self._cb_file_saved)
        self._sub_impact_timeout = mbb.SimpleEventSubscriber(self._cb_impact_timeout)

        self._sub_continue_click = mbb.SimpleEventSubscriber(self._cb_continue_click)
        self._sub_back_click = mbb.SimpleEventSubscriber(self._cb_back_click)
        self._sub_start_click = mbb.SimpleEventSubscriber(self._cb_start_test)
        self._sub_open_click = mbb.SimpleEventSubscriber(self._cb_open_result)

        # Connect events:
        self.gui.event_continue.connect(self._sub_continue_click)
        self.gui.event_back.connect(self._sub_back_click)
        self.gui.event_start.connect(self._sub_start_click)
        self.gui.ribbon.btn_open.event_click.connect(self._sub_open_click)

    @staticmethod
    def create_xml_settings():
        """
        :return: Standard XML settings for EMAWrapper
        """
        root = ET.Element('EMA_GUI_settings')
        ET.SubElement(root, 'n_reps').text = str(3)
        ET.SubElement(root, 'dir').text = './'

        # Test configuration:
        conf = mexp.ModalTestConfiguration('Modal test')
        root.append(conf.to_xml())

        # Spectral settings
        frf_sets = mfrf.SpectralAnalyzerConfiguration(i_input=3, i_output=1, sampling_rate=conf.sampling_rate,
                                                      filter_window=('exponential', None, 1e6),
                                                      window_fraction=0.5, overlap_fraction=0.5)
        root.append(frf_sets.to_xml())
        return root

    def _cb_start_test(self, _):
        # Show cylinder_properties form and ask to enter data
        self.task_thrd = thrd.Thread(target=self._test_cycle)
        self.task_thrd.start()

    def _test_cycle(self):

        # Get tool info:
        self.gui.set_title('New Modal Test')
        self.gui.show_toolset_info_form()
        self.wait_for_continue()

        tool_info = self.gui.get_tool_info()
        cases = []
        id_up = 0
        id_low = 0
        for key, (_id, use) in sorted(tool_info.items())[::-1]:
            if use == 1:
                cases.append(key)
            if key == 'upper':
                id_up = _id
            elif key == 'lower':
                id_low = _id

        # Connect ni_task to event subscribers:
        self._modal_test.event_file_saved.connect(self._sub_file_saved)
        self._modal_test.event_impact_request.connect(self._sub_impact_request)
        self._modal_test.event_impact_timeout.connect(self._sub_impact_timeout)
        self._modal_test.event_impact_detected.connect(self._sub_impact_detected)

        # Perform test
        for case in cases:
            self.gui.set_title('{0} cylinder_properties'.format(case.title()))
            # Update test description
            self._current_location = case
            self._modal_test.task_configuration.test_description = ''

            # Request place sensor
            self.gui.show_sensor_placement(case)
            self.wait_for_continue()

            for r in range(self._n_reps):

                # Set ni_task variables:
                repeat = True
                repeat_str = ''
                while repeat:
                    repeat = False
                    fn = '{5}{0}_IDU{1}_IDL{2}_{3}_R{4}.csv'.format(
                        datetime.datetime.today().strftime('%Y%m%d'), id_low, id_up, case, r+1, self._dir)

                    # Perform test
                    self.gui.set_title('{0} cylinder_properties, test {1}/{2} {3}'.format(case.title(), r+1, self._n_reps,
                                                                               repeat_str))
                    self._modal_test.do_test(fn)
                    self._modal_test.wait_until_done()

                    # Show results
                    self.gui.show_test_result(self._modal_test.log)
                    if self.wait_for_click() > 1:
                        repeat = True
                        repeat_str = '(retry)'

        # Disconnect event subscribers:
        self._modal_test.event_file_saved.disconnect(self._sub_file_saved)
        self._modal_test.event_impact_request.disconnect(self._sub_impact_request)
        self._modal_test.event_impact_timeout.disconnect(self._sub_impact_timeout)
        self._modal_test.event_impact_detected.disconnect(self._sub_impact_detected)

        self.gui.show_finished()
        self.gui.set_title('')

    def wait_for_click(self):
        """
        :return: 1 if continue clicked, 2 if back clicked, 3 if both clicked
        """
        self._continue_clicked = False
        self._back_clicked = False
        while not (self._continue_clicked or self._back_clicked):
            time.sleep(0.1)

        return int(str(self._continue_clicked + self._back_clicked*10), 2)

    def wait_for_continue(self):
        self._continue_clicked = False
        while not self._continue_clicked:
            time.sleep(0.1)

    def wait_for_back(self):
        self._back_clicked = False
        while not self._back_clicked:
            time.sleep(0.1)

    def _cb_open_result(self, _):
        """ Open test results

        :return:
        """
        fn = tk.filedialog.askopenfilename()
        if fn != '':
            self.gui.show_test_result(EMAWrapper.load_test_result(fn))
            self.gui.set_title("Opened test results")
            self.gui.set_message('Filename: {0}'.format(fn))

    @staticmethod
    def load_test_result(filename):
        """"Load experimental results corresponding to filename"""

        info, data = mlog.CSVLogReader().read(filename)
        log = mlog.Log(info)
        log.data = data
        return log

    def _cb_task_loaded(self, publisher):
        job_result = publisher.get_data()
        self._modal_test = job_result.data

        # Check result
        if type(self._modal_test) is mexp.ModalTest:
            frame_name = 'Modal Test'
        else:
            frame_name = 'SIMULATED Modal Test'

        self.gui.winfo_toplevel().title(frame_name)

        self.gui.set_state(GuiStates.IDLE)

    def _cb_waiting_for_impact(self, _):
        self.gui.show_impact_request(self._current_location)

    def _cb_impact_detected(self, _):
        self.gui.show_impact_detected(self._current_location)

    def _cb_impact_timeout(self, _):
        self.gui.show_impact_timeout()

    def _cb_file_saved(self, _):
        self.gui.set_message('File saved')

    def _cb_back_click(self, _):
        self._back_clicked = True

    def _cb_continue_click(self, _):
        self._continue_clicked = True


class BtnsSaveCancel(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.save = tk.Button(self, text='Save')
        self.save.pack(side=tk.RIGHT)
        self.cancel = tk.Button(self, text='Cancel')
        self.cancel.pack(side=tk.RIGHT)


class ConfigurationFrame(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(padx=5)

        # DAQ Settings:
        daq_info_dict = {'Chassis name': 'MadernRD',
                         'Module name': 'MadernRDMod1',
                         'Hammer channel': 1,
                         'Accel channel': 0,
                         'Sampling rate': 25600
                         }
        self._daq_info = self.get_labeled_frame('Experiment Info', daq_info_dict, 0, 0)

        # Experimental settings:
        exp_info_dict = {'Number of impacts': 3,
                         'Trigger value': 1000,
                         'Log size': 51200,
                         'Trigger timeout': 20,
                         'Test timeout': 20,
                         }
        self._exp_info = self.get_labeled_frame('Experiment Info', exp_info_dict, 0, 1)

        # Visualization Settings
        vis_settings = {'Number of impacts': 3,
                        'Trigger value': 1000,
                        'Trigger timeout': 20,
                        'Log size': 51200,
                        }
        self._vis_info = self.get_labeled_frame('Visualization Settings', vis_settings, 0, 2)

        self._btns = BtnsSaveCancel(self)
        self._btns.grid(column=0, row=3, sticky=tk.E)

    def get_labeled_frame(self, name, item_dict, column, row):
        frame = tk.LabelFrame(self, text=name, padx=5, pady=5)
        frame.grid(column=column, row=row, sticky=tk.W)

        for key, value in item_dict.items():
            item = mgui.LabeledEntry(master=frame, label=key, entry_conf={'width': 15})
            item.columnconfigure(0, minsize=120)
            item.value = value

        return frame


if __name__ == "__main__":

    ema = EMAWrapper(simulate=False, xml_filename='../gui_config/settings_ema_gui_sim.xml')
    ema.gui.mainloop()





