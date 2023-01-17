import traitlets, os, re, datetime, asyncio
from PySide2.QtWidgets import QMessageBox, QFileDialog
import threading
from madernpytools.qtgui.ui_project_log_widget import  Ui_ProjectLogWidget
from madernpytools.qtgui.traitlet_widgets import AbstractWidget
from madernpytools.backbone import HasTraitLinks
from madernpytools.log import Log, LogInfo, AbstractLog


class ProjectLogWidget(AbstractWidget, Ui_ProjectLogWidget, HasTraitLinks):
    log = AbstractLog()

    def __init__(self, parent):
        super().__init__(parent=parent)
        self._log = Log(log_info=LogInfo(description='', sampling_rate=1, signal_header=[]))
        self._log.observe(self._log_active_change, 'active')

        # Connect buttons
        self.btn_stop.clicked.connect(self._action_stop)
        self.btn_start.clicked.connect(self._action_start)
        self.btn_save.clicked.connect(self._action_save)

        self._store_dir = os.path.expanduser('~/vibration_measurements/')
        if not os.path.exists(self._store_dir):
            os.mkdir(self._store_dir)

    @traitlets.observe('log')
    def _log_change(self, change):
        if isinstance(change.old, AbstractLog):
            change.old.unobserve(self._log_active_change, 'active')
        if isinstance(change.new, AbstractLog):
            change.new.observe(self._log_active_change, 'active')

    def _action_stop(self):
        # Store result
        self.log.active = False

    def _action_start(self):
        # Check if dir exists:
        if self.log.n_samples > 0:
            res = QMessageBox.question(self, 'Data found', 'Omit existing data?',
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if QMessageBox.Yes == res:
                self.log.reset()
            if res != QMessageBox.Cancel:
                self.log.active = True
        else:
            self.log.active = True

    def _action_save(self):
        # Create filename:
        if self.log.n_samples > 0:
            time = datetime.datetime.now().strftime("%Y%M%d_%H%M%S")
            fn = f'{self._store_dir}{time}_{self.le_project.value}_vibr_M{self.le_maleid.value}_F{self.le_femaleid.value}.csv'

            # Store log
            self._t = threading.Thread(target=self._save_log, args=(self.log.data, self.log.info, fn))
            self._t.start()

            res = QMessageBox.information(self, 'Saving data', f'Saving logged data to:\n{fn}')

    def _save_log(self, data, log_info, fn):
        log = Log(log_info=log_info)
        log.data = data
        log.save(fn)

    def _log_active_change(self, change):
        if change.new:
            self.btn_save.setEnabled(False)
            self.btn_start.setEnabled(False)
        else:
            self.btn_save.setEnabled(True)
            self.btn_start.setEnabled(True)


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
    wid = ProjectLogWidget(parent=mainWindow)

    def add_samples(change):
        wid.log.add_sample(np.random.randn(10))
    wid.observe(add_samples, 'active')
    hlayout.addWidget(wid)
    mainWindow.resize(wid.size())

    # Show window
    mainWindow.show()
    sys.exit(app.exec_())


