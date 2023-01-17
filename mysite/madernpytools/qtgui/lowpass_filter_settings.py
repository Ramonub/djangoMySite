import traitlets

from PySide2.QtCore import Qt, Signal
from madernpytools.qtgui.ui_lowpass_filter_settings import Ui_LowPassFilterSettings
from madernpytools.qtgui.traitlet_widgets import AbstractWidget
from madernpytools.tools.frequency_response import LowPassFilter
from madernpytools.backbone import HasTraitLinks


class LowPassFilterSettings(AbstractWidget, Ui_LowPassFilterSettings, HasTraitLinks):
    filter = LowPassFilter(fs=1.0, order=2, low_pass_frequency=0.1)
    _value_signal = Signal(Qt.CheckState)

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.filter = LowPassFilter(fs=1.0, order=2, low_pass_frequency=0.1)

        self._trait_links = [
                              traitlets.link((self.filter, 'fs'), (self.sp_sampling_frequency, 'value')),
                              traitlets.link((self.filter, 'low_pass_frequency'), (self.sp_lowpass_frequency, 'value')),
                              traitlets.link((self.filter, 'order'), (self.sp_order, 'value'))
                             ]

        self.cb_overrule.observe(self._overrule_change, 'value')
        self.sp_sampling_frequency.setEnabled(False)

        self.gb_lp_settings.setChecked(self.filter.enabled)
        self.gb_lp_settings.toggled.connect(self._enabled_changed)
        self._value_signal.connect(self.gb_lp_settings.setChecked)

    def _overrule_change(self, change):
        self.sp_sampling_frequency.setEnabled(change['new'])

    @traitlets.observe('filter')
    def _filter_change(self, change):
        if isinstance(change['old'], LowPassFilter):
            change['old'].unobserve(self._enabled_change, 'enabled')
            change['old'].unobserve(self._fs_change, 'fs')
        if isinstance(change['new'], LowPassFilter):
            # Relink:
            self.update_links(src_obj=change.new)

            # We could consider to update these values:
            self.gb_lp_settings.setChecked(self.filter.enabled)
            change['new'].observe(self._enabled_change, 'enabled')
            change['new'].observe(self._fs_change, 'fs')

    def _set_new_filter_values(self, filter: LowPassFilter):
        if filter.fs < self.sp_sampling_frequency.value:
            # First set (decrease) low-pass frequency to prevent potential invalid low-pass filter settings:
            self.sp_lowpass_frequency.value = filter.low_pass_frequency
            self.sp_sampling_frequency.value = filter.fs
        else:
            # First increase sampling frequency:
            self.sp_sampling_frequency.value = filter.fs
            self.sp_lowpass_frequency.value = filter.low_pass_frequency

        # Setup order:
        self.sp_order.value = filter.order
        self._set_lp_max()

    def _set_lp_max(self):
        corr = 10**-(self.sp_lowpass_frequency.decimals())
        self.sp_lowpass_frequency.setMaximum(self.filter.fs/2 - corr)

    def _fs_change(self, change):
        self._set_lp_max()

    def _enabled_change(self, change):
        if change['new']:
            self._value_signal.emit(Qt.Checked)
        else:
            self._value_signal.emit(Qt.Unchecked)

    def _enabled_changed(self, value):
        self.filter.enabled = self.gb_lp_settings.isChecked()


if __name__ == "__main__":

    import sys
    from PySide2.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QSizePolicy

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    hlayout = QHBoxLayout(mainWindow.centralWidget())

    # Define widget
    wid = LowPassFilterSettings(parent=mainWindow)
    mainWindow.resize(wid.size())
    hlayout.addWidget(wid)

    mainWindow.show()
    sys.exit(app.exec_())
