from madernpytools.qtgui.traitlet_widgets import QTraitCheckBox, AbstractWidget
from scancontrol.profile_filters import ISignalKeyList, SignalKeyList
from madernpytools.qtgui.ui_signal_list import Ui_SignalList
import traitlets
import warnings


class SignalList(AbstractWidget, Ui_SignalList):
    selected_items = SignalKeyList(default_value=SignalKeyList())
    items = SignalKeyList(default_value=SignalKeyList())

    def __init__(self, parent):
        super().__init__(parent=parent)
        self._checkboxes = {}
        self._ignore_update = False

    @traitlets.observe('selected_items', 'items')
    def _selecteditems_changed(self, change):
        # Check if selected items are in items:
        if not all([item in self.items for item in self.selected_items]):
            # Update selected items:
            self.selected_items = [item for item in self.selected_items if item in self.items]

        if not self._ignore_update:
            self._load_items()

    def _load_items(self):
        """

        :param items: List of items to generate checkbox
        :param selected_items:  List of items to check (should be subsection of items)
        :return:
        """
        # Clear items:
        for cb, item in self._checkboxes.items():
            self.verticalLayout.removeWidget(cb)              # Remove from layout
            self.scrollAreaWidgetContents.children()
            cb.unobserve(self._cb_value_changed, 'value')     # Unobserve value
            cb.setVisible(False)
            cb.deleteLater()
        self._checkboxes.clear()

        # Add new items:
        for i, item in enumerate(sorted(self.items)):
            # Get name:
            cb = QTraitCheckBox(parent=self.scrollAreaWidgetContents)
            cb.setText(str(item))

            # Get items:
            if item in self.selected_items:
                cb.value = True
            cb.observe(self._cb_value_changed, 'value')

            self.verticalLayout.insertWidget(i, cb)
            self._checkboxes[cb] = item

    def _cb_value_changed(self, change):
        # Update selected items:
        self._ignore_update = True
        self.selected_items = [item for cb, item in self._checkboxes.items() if cb.value]
        self._ignore_update = False

    def load_items(self, items: list, selected_items: list = None):
        """ Load items

        :param items:
        :param selected_items:
        :return:
        """
        warnings.warn('load_items() will be removed in future versions, ' +
                      'directly assign items and selected_items attribute instead',
                      DeprecationWarning)

        self.items = SignalKeyList(items)
        if selected_items is None:
            self.selected_items = SignalKeyList([])
        else:
            self.selected_items = SignalKeyList(selected_items)

    def get_selected_items(self):
        """ Get selected items

        :return:
        """
        warnings.warn('load_items() will be removed in future versions, ' +
                      'directly assign items and selected_items attribute instead',
                      DeprecationWarning)
        return self.selected_items


if __name__ == "__main__":

    import sys
    from PySide2.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QSizePolicy

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    hlayout = QHBoxLayout(mainWindow.centralWidget())

    def save_test():
        print(wid.get_selected_signals())

    wid = SignalList(parent=mainWindow)
    hlayout.addWidget(wid)
    mainWindow.resize(wid.size())

    mainWindow.show()
    wid.load_items(['a', 'b', 'c', 'd'])
    sys.exit(app.exec_())
