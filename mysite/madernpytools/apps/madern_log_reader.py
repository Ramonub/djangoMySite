import tkinter as tk
import os
import tkinter.ttk as ttk
import numpy as np
from PIL import Image, ImageTk
import madernpytools.qtgui as mgui
import madernpytools.log as mlog
import madernpytools.backbone as mbb
import madernpytools.tools.frequency_response as mfrf

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class RibbonItem(tk.Button):

    def __init__(self, master, icon_fn='', **kwargs):
        """
        
        :param master:
        :param icon_fn:
        :param kwargs:
        """

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

        self.btn_open = RibbonItem(self, text='Open...')
        self.btn_open.pack(side=tk.LEFT)

    def set_state(self, state=tk.NORMAL):
        self.btn_open.configure(state=state)


class DoubleCheckbox(tk.Frame):

    def __init__(self, master=None, label=''):
        tk.Frame(self, master=master)
        self._signal_chk = tk.Checkbutton(self, text=item, command=self._value_click, variable=self._var)

        self._xvar = tk.IntVar()
        self._yvar = tk.IntVar()

        self._x_box = tk.Checkbutton(self, text='', command=self._x_click, variable=self._xvar)
        self._y_box = tk.Checkbutton(self, text='', command=self._y_click, variable=self._yvar)
        self._label = tk.Label(master=master, text=label)

        self._x_box.pack()
        self._y_box.pack()
        self._label.pack()

        self.event_x_clicked = mbb.EventPublisher(data_type=DoubleCheckbox)
        self.event_y_clicked = mbb.EventPublisher(data_type=DoubleCheckbox)

    def _x_click(self):
        self.event_x_clicked.raise_event(self)

    def _y_click(self):
        self.event_y_clicked.raise_event(self)


class AxisItemSelection(tk.Frame):

    def __init__(self, master=None, item_list=None):

        tk.Frame.__init__(self, master=master)
        self.grid()

        self.event_checked = mbb.EventPublisher(data_type=mgui.ItemCheckBox)
        self._click_subscriber = mbb.SimpleEventSubscriber(self._cb_clicked)
        self.check_boxes = []

        self.load(item_list)

    def _cb_clicked(self, publisher):
        item = publisher.get_data()
        self.event_checked.raise_event(publisher.get_data())

    def load(self, item_list):
        item_list = mbb.ArgumentVerifier(list, []).verify(item_list)

        for cb in self.check_boxes:
            cb.destroy()
        self.check_boxes=[]

        i=0
        for item in item_list:
            val = DoubleCheckbox(item, self)
            val.event_x_clicked.connect(self._click_subscriber)
            val.event_y_clicked.connect(self._click_subscriber)
            val.grid(column=0, row=i, sticky=tk.W)
            self.check_boxes.append(val)
            i+=1


class ItemSelection(tk.Frame):

    def __init__(self, master=None, item_list=None):

        tk.Frame.__init__(self, master=master)
        self.grid()

        self.event_checked = mbb.EventPublisher(data_type=mgui.ItemCheckBox)
        self._click_subscriber = mbb.SimpleEventSubscriber(self._cb_clicked)
        self.check_boxes = []

        self.load(item_list)

    def _cb_clicked(self, publisher):
        item = publisher.get_data()
        self.event_checked.raise_event(publisher.get_data())

    def load(self, item_list):
        item_list = mbb.ArgumentVerifier(list, []).verify(item_list)

        for cb in self.check_boxes:
            cb.destroy()
        self.check_boxes=[]

        i=0
        for item in item_list:
            val = mgui.ItemCheckBox(item, self)

            val.event_checked.connect(self._click_subscriber)
            val.grid(column=0, row=i, sticky=tk.W)
            self.check_boxes.append(val)
            i+=1


class ToolBoxedFigure(tk.Frame):

    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master=master)

        # Setup figure:
        self._gui_fig = mgui.GUIPlot(self)
        self._gui_fig.grid()

        self.ax = self.fig.add_subplot(111)
        self.ax.grid()
        self._lines = {}


    @property
    def fig(self):
        return self._gui_fig.fig

    def show_navigationbar(self, value):
        if value:
            self._navbar = NavigationToolbar2Tk(self._canvas, self)
        else:
            self._navbar.destroy()

    def add(self, data, key):
        if not key in self._lines:
            l,= self.ax.plot(data[0], data[1], label=key)
            self._lines[key] = l
            self.ax.legend()
            self.ax.relim()
            self._gui_fig.refresh()

    def remove(self, key):
        if key in self._lines:
            l = self._lines.pop(key)
            l.remove()
            if len(self._lines)==0:
                self.ax.legend_.remove()
            else:
                self.ax.legend()
            self.ax.relim()
            self._gui_fig.refresh()


class BoschGui(tk.Frame):

    def __init__(self, master=None, **kwargs):

        tk.Frame.__init__(self, master)
        self.grid()
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        ##----- GUI Elements
        # Ribbon
        self._ribbon = RibbonFrame(self)
        self._ribbon.grid(column=0, row=0, columnspan=2)

        # Selection:
        self._label_frame = tk.LabelFrame(self, text='Signals')
        self._label_frame.grid(column=0, row=1)
        self._item_selection = ItemSelection(self._label_frame, item_list=[])

        #x-axis:
        self._x_sel = ttk.Combobox(self, state='readonly', width=40)
        self._x_sel.grid(column=1, row=2)
        self._x_sel.bind('<<ComboboxSelected>>', self._cb_x_axis_changed)

        # Figures
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(column=1,row=1)
        self._time_fig = ToolBoxedFigure(self)
        self._frf_fig = ToolBoxedFigure(self)
        self._frf_fig.ax.set_xlabel('Frequency [Hz]')

        self._notebook.add(self._time_fig, text='Time sequence')
        self._notebook.add(self._frf_fig, text='Frequency response')

        # BoschLog
        self._log = None

        # Event subscribers:
        self._sub_open = mbb.SimpleEventSubscriber(h_callback=self._cb_open_file)
        self._sub_item_checked = mbb.SimpleEventSubscriber(h_callback=self._cb_item_checked)

        # Connect events:
        self._ribbon.btn_open.event_click.connect(self._sub_open)
        self._item_selection.event_checked.connect(self._sub_item_checked)

    def _cb_open_file(self, publisher):
        fn = tk.filedialog.askopenfilename()
        if fn != '':
            if os.path.splitext(fn)[-1]=='.txt':
                # Bosch Log
                self._log = mlog.BoschLogReader().read(fn)

            elif os.path.splitext(fn)[-1]=='.csv':
                # Siemens log:
                self._log = mlog.SiemensLogReader().read(fn)

            # Update GUI items:
            self._item_selection.load(self._log.info.signal_header)
            self._x_sel.config(values=self._log.info.signal_header)

            self._time_fig.ax.set_title(os.path.split(fn)[-1])
            self._frf_fig.ax.set_title(os.path.split(fn)[-1])

    def _cb_x_axis_changed(self, event):
        # Value changed
        x_key = event.widget.get()
        self._time_fig.ax.set_xlabel(x_key)
        self._time_fig.update()

        if x_key != "":
            for check_box in self._item_selection.check_boxes:
                # Clear fig:
                self._time_fig.remove(check_box.item)

                # Add if checked:
                if check_box.state:
                    y_key = check_box.item
                    my_thread = mbb.thrd.Thread(target=self._load_data, args=[x_key, y_key])
                    my_thread.start()

    def _load_data(self, x_key, y_key):

        data = self._log.get_signal_pair(x_key=x_key, y_key=y_key)
        #y_data = self._log.get_signal(key, include_time=False)
        #x_data = self._log.get_signal(self._x_sel.get(), include_time=False)
        x_data = data[0]
        y_data = data[1]

        # Time-based plots
        self._time_fig.add([x_data, y_data], y_key)

        # FrF
        self._frf_fig.add(self._perform_frf(self._log.get_signal(y_key)), y_key)

    def _cb_item_checked(self, publisher):
        checkbox = publisher.get_data()
        print('{0} checked: {1}'.format(checkbox.item, checkbox.state))

        key = checkbox.item
        if checkbox.state:
            x_key = self._x_sel.get()
            y_key = key

            if x_key != "" and y_key != "":
                my_thread = mbb.thrd.Thread(target=self._load_data, args=[x_key, y_key])
                my_thread.start()
        else:
            self._time_fig.remove(key)
            self._frf_fig.remove(key)

    def _perform_frf(self, data):

        # Estimate sampling rate:
        dt = np.gradient(data[0]).mean()
        sampling_rate = int(1/(dt*1e-3))

        # Frequency analysis:
        conf = mfrf.SpectralAnalyzerConfiguration(f_range=[0, sampling_rate/2],
                                                  i_input=0,
                                                  i_output=None,
                                                  sampling_rate=sampling_rate,
                                                  filter_window='hann',
                                                  window_fraction=0.1, overlap_fraction=0.1)
        analyzer = mfrf.SpectralAnalyzer(conf)
        analyzer.configuration.i_output = None
        res = analyzer.analyze(data[1][:, None])

        return res.frequencies, np.log10(np.abs(res.Sff))



if __name__ == "__main__":

    mygui = BoschGui()
    mygui.mainloop()

