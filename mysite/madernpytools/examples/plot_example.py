import traitlets
from PySide2.QtWidgets import QApplication, QMainWindow
import numpy as np
import sys, threading, time
from matplotlib.lines import Line2D
import madernpytools.tools.utilities as mutils
from madernpytools.qtgui.mplfigure import MplFigure
from madernpytools.signal_handling import Buffer



class SomeGenerator(traitlets.HasTraits):   # Indicates that we implement traits in this class
    """ A generator class to generate some random data

    """
    output = traitlets.TraitType()            # Our output signal

    def __init__(self, rate=1):
        super().__init__(output=mutils.ListofDict([])  # Initialize 'output'
                         )   # Initialize base classe

        self._rate = rate
        self._keep_running = True

    def start_generation(self):
        """ Start data generation

        :return:
        """
        self._keep_running = True
        self._thrd = threading.Thread(target=self._worker)
        self._thrd.start()

    @property
    def sampling_rate(self):
        return self._rate

    def stop_generation(self):
        """ Stop data generation

        :return:
        """
        self._keep_running = False

    def _worker(self):
        t = 0
        while self._keep_running:
            self.output = mutils.ListofDict([dict(zip(['x', 'y'], [t, np.sin(0.1*t)]))])
            t+=1
            time.sleep(self._rate**-1)


if __name__ == "__main__":
    """
    Here we demonstrate how to:
    * Generate data (using the class above)
    * Link it to a buffer,
    * Plot the buffer using MplFigure in a QT-window
    """

    # Define window and figure:
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()

    qfig = MplFigure(parent=mainWindow)
    mainWindow.resize(qfig.size())

    # Setup axis:
    ax = qfig.figure.gca()

    # Add a line:
    line = Line2D([], [], marker='.', ms=5, label='Some Line')
    ax.add_line(line)

    # Make axis look nice
    ax.grid(True)
    ax.set_xlabel('X data')
    ax.set_ylabel('Y data')
    ax.legend()

    # Data generation and display:
    def buffer_change(change):
        """ A function which handles buffer changes

        :param change:
        :return:
        """
        if isinstance(change.new, mutils.Listof):  # Ensure that we have a ListOf
            x = change.new.get_key_values('x')         # Get  data
            y = change.new.get_key_values('y')
            line.set_data(x, y)                    # Set to line

            # Update axis:
            ax.relim()
            ax.autoscale_view()

            qfig.refresh()

    gen = SomeGenerator(rate=20)
    buffer = Buffer(n=100)
    traitlets.link((gen, 'output'), (buffer, 'input_data'))
    buffer.observe(buffer_change, 'output_data')
    gen.start_generation()

    mainWindow.setCentralWidget(qfig)
    mainWindow.show()
    sys.exit(app.exec_())




