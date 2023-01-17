from PySide2.QtWidgets import  QApplication, QMainWindow,  QWidget
from PySide2.QtGui import QShowEvent
from PySide2.QtCore import Signal
from madernpytools.qtgui.traitlet_widgets import AbstractWidget
from madernpytools.qtgui.ui_mplfigure import Ui_MplFigure
from madernpytools.plot import ResetableFuncAnimation
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import asyncio, logging, PySide2

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure

_logger = logging.getLogger(f'madernpytools.{__name__}')


class MplFigure(AbstractWidget, Ui_MplFigure):

    def __init__(self, parent):
        super().__init__(parent=parent)
        self._fig = Figure(tight_layout=True)
        self._setup_canvas()

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.show_toolbar()

        # Handling of the Animation function should be Thread safe. I experienced 'Timers cannot be started from another
        # thread' errors when starting/stopping the event_source directly. To avoid these warnings, we define signals
        # for start/stop/reset. These signals ensure that the CanvasTimer is used from the QT-GUI Thread.

    @property
    def figure(self):
        return self._fig

    def _setup_canvas(self):
        self.canvas = FigureCanvas(self._fig)
        self.mplvl.addWidget(self.canvas)
        res = self.palette().color(self.backgroundRole())
        self.figure.patch.set_facecolor([res.redF(), res.greenF(), res.blueF()])
        #self.figure.patch.set_alpha(1.0)
        self.canvas.draw()

    def refresh(self):
        """ Refresh figure

        @return:
        """
        self.canvas.draw()

    def show_toolbar(self):
        # add toolbar
        self.canvas.toolbar_visible = True

    def hide_toolbar(self):
        # add toolbar
        self.canvas.toolbar_visible = False


class AnimatedMplFigure(AbstractWidget, Ui_MplFigure):

    _reset_signal = Signal()
    _start_signal = Signal()
    _stop_signal = Signal()

    def __init__(self, parent, animation_interval=250, blit=True):
        super().__init__(parent=parent)
        self._fig = Figure(tight_layout=True)
        self._setup_canvas()

        # Handling of the Animation function should be Thread safe. I experienced 'Timers cannot be started from another
        # thread' errors when starting/stopping the event_source directly. To avoid these warnings, we define signals
        # for start/stop/reset. These signals ensure that the CanvasTimer is used from the QT-GUI Thread.
        self._reset_signal.connect(self._reset)
        self._start_signal.connect(self._start_animation)
        self._stop_signal.connect(self._stop_animation)

        self._artists = []
        self._animation = ResetableFuncAnimation(fig=self._fig, func=self._animation_func, interval=animation_interval,
                                                 blit=blit)

        self._reset_cnt = 0

    def _reset(self, *args):
        """ Reset animation

        @param args:
        @return:
        """
        self._reset_cnt +=1
        #print(f'Reset call {id(self)} {self._reset_cnt}')
        self._animation.reset()

    def _stop_animation(self, *args):
        """ Stop animaton event source

        @param args:
        @return:
        """
        #print(f'Stop call {id(self)} {self._reset_cnt}')
        self._animation.event_source.stop()

    def _start_animation(self):
        #print(f'Start call {id(self)} {self._reset_cnt}')
        self._animation.event_source.start()

    @property
    def animation(self):
        return self._animation

    def reset(self):
        self._reset_signal.emit()

    def showEvent(self, event:PySide2.QtGui.QShowEvent) -> None:
        """ Responds to Figures show event, and causes any animated artists to start

        @param event:
        @return:
        """

        # Emit a signal to start event_source
        self._start_signal.emit()

    def hideEvent(self, event:PySide2.QtGui.QShowEvent) -> None:
        """ Responds to Figures hide event, and causes any animated artists to pause

        @param event:
        @return:
        """
        # Emit a signal to stop event_source
        self._stop_signal.emit()

    @property
    def animated_artists(self):
        if self.isVisible():
            return self._artists
        else:
            print('Ignore update')
            return []

    def add_animated_artist(self, art):
        """
        Add an artist to be handled by animation.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)
        self.reset()

    def remove_animated_artist(self, art):
        """Removes animated artist from animation an resets animation"""
        self._artists.remove(art)
        self.reset()

    def _animation_func(self, *args):
        """ Function callback for animation function

        @param args:
        @return: animated artists
        """
        return self._artists

    @property
    def figure(self):
        return self._fig

    def _setup_canvas(self):
        self.canvas = FigureCanvas(self._fig)
        self.mplvl.addWidget(self.canvas)
        res = self.palette().color(self.backgroundRole())
        self.figure.patch.set_facecolor([res.redF(), res.greenF(), res.blueF()])
        #self.figure.patch.set_alpha(1.0)
        self.canvas.draw()

    def refresh(self):
        """ Refresh figure

        @return:
        """
        #self.canvas.draw()
        #pass
        self.reset()


if __name__ == "__main__":
    import sys, threading, time
    import numpy as np
    from matplotlib.lines import Line2D

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()

    qfig = AnimatedMplFigure(parent=mainWindow)
    mainWindow.resize(qfig.size())

    # Setup axis:
    ax = qfig.figure.gca()
    ax.grid(True)

    mainWindow.setCentralWidget(qfig)
    mainWindow.show()

    lines = []

    def update_func():
        t=0
        axis_set=False
        while True:
            for i, line in enumerate(lines):
                x = np.linspace(0, np.pi * 2, 100)
                y = np.sin(x + t*0.1 + np.pi/4*i)
                line.set_data(x,y)

                if not axis_set:
                    axis_set = True
                    ax.relim()
                    ax.autoscale_view()
            #    qfig.refresh_animated_artists()
            time.sleep(0.01)
            #await asyncio.sleep(0.01)

            t+=1

    def _mouse_clicked(event):

        if event.button==1:
            lines.append(Line2D([], [], lw=len(lines)+1, label=f'Line {len(lines)}'))
            ax.add_line(lines[-1])
            qfig.add_animated_artist(lines[-1])
        if event.button==3 and len(lines) > 0:
            qfig.remove_animated_artist(lines.pop())

    thrd = threading.Thread(target=update_func)
    thrd.start()

    qfig.canvas.mpl_connect('button_release_event', _mouse_clicked)

    sys.exit(app.exec_())

