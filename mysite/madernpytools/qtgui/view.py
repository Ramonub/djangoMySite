
import sys

from PySide2.QtWidgets import QMainWindow, QApplication
from madernpytools.qtgui.ui_spin_test import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):

        QMainWindow.__init__(self)
        self.setupUi(self)

        # Observe
        for obj in [self.spinBox, self.doubleSpinBox, self.lineEdit, self.verticalSlider, self.horizontalSlider,
                    self.checkBox]:
            obj.observe(self.value_change, names='value')

    def value_change(self, change):
        print('Got a change from {}: {}'.format(change['owner'].objectName(), change['new']))


if __name__ == '__main__':

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())






