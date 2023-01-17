import traitlets, os
from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import QFileDialog
from madernpytools.qtgui.traitlet_widgets import AbstractWidget
from madernpytools.qtgui.ui_filebrowser import Ui_FileBrowser


class FileBrowser(AbstractWidget, Ui_FileBrowser):
    path = traitlets.CUnicode(help='File path', default_value='')

    def __init__(self, parent):
        super().__init__(parent=parent)

        self.pb_browse.clicked.connect(self._browse_click)

        self.caption = 'Please select file'
        self.file_filter = ''
        self.dir = "~/"

        traitlets.link((self, 'path'), (self.le_path, 'value'))

    def _browse_click(self):
        filename = QFileDialog.getOpenFileName(self, caption=self.file_filter, dir=self.dir,
                                               filter=self.file_filter)[0]
        if filename != '':
            self.dir = os.path.dirname(filename)
            self.path = filename


if __name__ == "__main__":

    import sys
    from PySide2.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QSizePolicy

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    hlayout = QHBoxLayout(mainWindow.centralWidget())
# Define widget
    wid = FileBrowser(parent=mainWindow)
    mainWindow.resize(wid.size())
    hlayout.addWidget(wid)

    def cb_path_change(change):
        print('Received: ', change.new)

    mainWindow.show()
    sys.exit(app.exec_())
