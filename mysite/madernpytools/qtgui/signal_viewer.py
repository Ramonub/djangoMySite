import traitlets, os, re, asyncio, logging
from matplotlib.lines import Line2D
from matplotlib import cm
import numpy as np

import madernpytools.backbone as mbb
from madernpytools.qtgui.ui_signal_viewer import Ui_SignalViewer
from madernpytools.qtgui.traitlet_widgets import AbstractWidget
import madernpytools.tools.utilities as mutils
from madernpytools.signal_handling import SignalKeyList

logger = logging.getLogger(f'madernpytools.{__name__}')
colors = cm.get_cmap('Set1')(np.linspace(0,1,10))


class SignalViewer(AbstractWidget, Ui_SignalViewer):
    input_data = mutils.ListofDict()
    x_key = traitlets.CUnicode(default_value='time')

    def __init__(self, parent):
        super().__init__(parent=parent)

        self.signal_selection.groupBox.setCheckable(False)

        self._ax = self.signal_display.figure.add_subplot(111)
        self._setup_figure()
        #self.signal_selection.observe(self._data_selector_change, 'data_selector')
        #self.signal_selection.observe(self._data_change, 'selected_items')
        asyncio.create_task(self._update_loop())

        self._line_dict = {}

    @traitlets.default('input_data')
    def _default_input(self):
        return mutils.ListofDict()

    @traitlets.observe('input_data')
    def _data_change(self, change):
        self._update_lines(self.input_data[:])
        logger.info('Received new data ')
        pass

    def _setup_figure(self):
        self._ax.grid(True)
        self._ax.set_xlabel(self.x_key)

    async def _update_loop(self):
        while True:
            await self._refresh_plot(self.input_data[:])
            await asyncio.sleep(0.2)

    @traitlets.observe('x_key')
    def _xkey_change(self, change):
        self._setup_figure()

    def refresh_plot(self, blocking=False):
        if not blocking:
            asyncio.create_task(self._refresh_plot(self.input_data[:]))
        else:
            self._refresh_plot(self.input_data[:])

    async def _refresh_plot(self, data: mutils.ListofDict):
        self.signal_display.refresh()

    def _update_lines(self, data: mutils.ListofDict):

        # Check if x_key is in data:
        if not self.x_key in data.get_keys():
            return

        # Remove non-selected lines:
        new_dict = {}
        for key, line in self._line_dict.items():
            if key in self.signal_selection.selected_items:
                new_dict[key] = line
            else:
                line.remove()
        self._line_dict = new_dict

        if len(data) > 10000:
            step = len(data)//10000
            data = data[::step]

        # Plot available keys:
        for key in self.signal_selection.selected_items:
            if key in data.get_keys():
                if key in self._line_dict.keys():
                    # Update existing line:
                    self._line_dict[key].set_data(data.get_key_values(self.x_key),
                                                  data.get_key_values(key))
                    self._line_dict[key].set_visible(True)
                else:
                    # Remove existing line:
                    i = len(self._line_dict) % len(colors)
                    self._line_dict[key] = Line2D(xdata=data.get_key_values(self.x_key),
                                                  ydata=data.get_key_values(key),
                                                  color=colors[i, :])
                    self._ax.add_line(self._line_dict[key])

        self._ax.autoscale()
        self._ax.relim()


if __name__ == "__main__":
    import sys
    import numpy as np
    from PySide2.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QSizePolicy

    # Define MainWindow & Layout:
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    hlayout = QHBoxLayout(mainWindow.centralWidget())

    # Define test widget:
    wid = SignalViewer(parent=mainWindow)
    hlayout.addWidget(wid)

    ax = wid.signal_display.figure.add_subplot(111)
    ax.plot(np.random.randn(10))
    wid.signal_display.refresh()

    # Show window
    mainWindow.resize(wid.size())
    mainWindow.show()
    sys.exit(app.exec_())



