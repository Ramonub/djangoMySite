import traitlets, os, re, logging
from PySide2.QtCore import QObject, Qt, Signal, Slot
from PySide2.QtWidgets import QMessageBox, QFileDialog
from PySide2.QtGui import QIcon
from madernpytools.qtgui.ui_autosave_log_widget import Ui_AutosaveLogWidget
from madernpytools.qtgui.traitlet_widgets import AbstractWidget
from madernpytools.backbone import HasTraitLinks
from madernpytools.log import AutoSaveLog, LogInfo

logger = logging.getLogger(f'madernpytools.{__file__}')


class AutosaveLogWidget(AbstractWidget, Ui_AutosaveLogWidget, HasTraitLinks):

    def __init__(self, parent):
        super().__init__(parent=parent)
        self._log = AutoSaveLog('', log_info=LogInfo(description='', sampling_rate=1, signal_header=[]))
        self._log.observe(self._log_active_change, 'active')

        # Connect buttons
        self.btn_stop.clicked.connect(self._action_stop)
        self.btn_start.clicked.connect(self._action_start)
        self.btn_select_dir.clicked.connect(self._action_select_dir)

        self.le_logdir.value = os.path.expanduser('~')

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, log: AutoSaveLog):
        self._log.unobserve(self._log_active_change, 'active')
        self._log = log
        self._log.observe(self._log_active_change, 'active')

    def _action_stop(self):
        logger.info('Stop logging request')
        self._log.autosave = False

    def _action_start(self):
        logger.info('Start logging request')
        # Check if dir exists:

        if os.path.exists(self.le_logdir.value):
            existing_files = self._existing_files(pattern='{0}\d{{6}}.csv')
            if len(existing_files) == 0:
                self._log.set_filename(f'{self.le_logdir.value}/{self.le_basefilename.value}')
                self._log.autosave = True
            else:
                if len(existing_files) > 10:
                    file_list = ''.join(' - {0},\n'.format(f) for f in existing_files[:5]) + '...\n' + \
                                ''.join(' - {0},\n'.format(f) for f in existing_files[-5:])
                else:
                    file_list = ''.join(' - {0},\n'.format(f) for f in existing_files)
                mes = "Files with selected filename already exist:" \
                      " \n{0}do you want to overwrite them?".format(file_list)
                qmes = QMessageBox(self, text=mes)
                qmes.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                qmes.setDefaultButton(QMessageBox.Yes)
                ret = qmes.exec_()
                if ret == QMessageBox.Yes:
                    self._log.set_filename(f'{self.le_logdir.value}/{self.le_basefilename.value}')
                    self._log.autosave = True
        else:
            mes = f'Selected directory not found: {self.le_logdir.value}'
            QMessageBox(self, text=mes, buttons=(QMessageBox.Yes, QMessageBox.NoButton)).show()

    def _log_active_change(self, change):
        if change.new:
            self.le_basefilename.setEnabled(False)
            self.le_logdir.setEnabled(False)
            self.le_description.setEnabled(False)
        else:
            self.le_basefilename.setEnabled(True)
            self.le_logdir.setEnabled(True)
            self.le_description.setEnabled(True)

    def _existing_files(self, pattern='{0}.csv'):
        sel_files = [f for f in os.listdir(self.le_logdir.value) if re.match(pattern.format(
            os.path.splitext(self.le_basefilename.value)[0]), f)]
        return sel_files

    def _action_select_dir(self):
        res = QFileDialog.getExistingDirectory(self, caption='Select directory to store measurement data',
                                         dir='~/')
        if res != '':
            self.le_logdir.value = res


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QSizePolicy

    # Define MainWindow & Layout:
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    hlayout = QHBoxLayout(mainWindow.centralWidget())

    # Define test widget:
    wid = AutosaveLogWidget(parent=mainWindow)
    hlayout.addWidget(wid)
    mainWindow.resize(wid.size())

    # Show window
    mainWindow.show()
    sys.exit(app.exec_())


