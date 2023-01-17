from madernpytools.qtgui.traitlet_widgets import QTraitCheckBox, AbstractWidget
from madernpytools.qtgui.ui_signal_selection import Ui_SignalSelection
from madernpytools.filtering import DataSelector
import traitlets


class SignalSelection(AbstractWidget, Ui_SignalSelection):
    data_selector = DataSelector()
    selected_items = traitlets.TraitType()
    available_items = traitlets.TraitType()

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.data_selector = DataSelector()

    @traitlets.observe('data_selector')
    def _selector_change(self, change):

        traitlets.link((change.new, 'available_keys'), (self.signal_list, 'items'))
        traitlets.link((change.new, 'selected_keys'), (self.signal_list, 'selected_items'))
        traitlets.link((change.new, 'selected_keys'), (self, 'selected_items'))
        traitlets.link((change.new, 'available_keys'), (self, 'available_items'))

    def load_items(self, items: list, selected_items: list=None):
        """

        :param items: List of items to generate checkbox
        :param selected_items:  List of items to check (should be subsection of items)
        :return:
        """
        self.signal_list.load_items(items, selected_items)

    def get_selected_items(self):
        """

        :return:
        """
        return self.signal_list.get_selected_items()


if __name__ == "__main__":

    import sys
    from PySide2.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QSizePolicy

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    hlayout = QHBoxLayout(mainWindow.centralWidget())

    def save_test():
        print(wid.get_selected_signals())

    wid = SignalSelection(parent=mainWindow)
    wid.load_items(['a', 'b', 'c', 'd'])
    hlayout.addWidget(wid)
    mainWindow.resize(wid.size())

    mainWindow.show()
    sys.exit(app.exec_())
