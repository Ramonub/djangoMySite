import traitlets, time
from madernpytools.backbone import IProcessStatus
from PySide2.QtWidgets import QWidget, QMainWindow, QDoubleSpinBox, QSpinBox, QLineEdit, QSlider, QComboBox, QCheckBox, \
    QDialog, QProgressBar, QLabel
from PySide2.QtCore import QObject, Qt, Signal, Slot
"""
To incorporate traitlets in Qt classes, we need to inherit both in a single class. Doing this straightforwardly will
give a metaclass error. 

The reason, traitlets and Qt Widgets don't share the same metaclass 
(http://www.phyast.pitt.edu/~micheles/python/metatype.html)

To resolve this, we need to define a new metaclass that inherits both metaclasses, and explicitly refer to the new metaclass
when inheriting both Qt-Widgets and HasTraits 

Below we do this for 

"""


class MetaTraitQObject(type(traitlets.HasTraits), type(QObject)):
    pass


class ITraitQWidget(traitlets.HasTraits, QWidget, metaclass=MetaTraitQObject):
    pass


class ITraitQDialog(traitlets.HasTraits, QDialog, metaclass=MetaTraitQObject):
    pass


class AbstractWidget(ITraitQWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)


class AbstractDialog(ITraitQDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)




class AbstractMainWindow(traitlets.HasTraits, QMainWindow, metaclass=MetaTraitQObject):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)


class QTraitSpinBox(traitlets.HasTraits, QSpinBox, metaclass=MetaTraitQObject):
    value = traitlets.CInt()
    _value_signal = Signal(int)

    def __init__(self, parent, PySide2_QtWidgets_QWidget=None, NoneType=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__
        super().__init__(parent=parent)
        self.valueChanged.connect(self.sp_value_changed)
        self._value_signal.connect(self.setValue)

        self._ignore_update = False

    @traitlets.observe('value')
    def value_change(self, change):
        if not self._ignore_update:
            self._value_signal.emit(int(change['new']))

    def sp_value_changed(self, value):
        self._ignore_update = True
        self.value = value
        self._ignore_update = False


class QTraitDoubleSpinBox(traitlets.HasTraits, QDoubleSpinBox, metaclass=MetaTraitQObject):
    value = traitlets.CFloat()
    _value_signal = Signal(float)

    def __init__(self, parent, PySide2_QtWidgets_QWidget=None, NoneType=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__
        super().__init__(parent=parent)
        self.valueChanged.connect(self.sp_value_changed)
        self._value_signal.connect(self.setValue)

        self._ignore_update = False

    @traitlets.observe('value')
    def value_change(self, change):
        if not self._ignore_update:
            self._value_signal.emit(float(change['new']))

    def sp_value_changed(self, value):
        self._ignore_update = True
        self.value = value
        self._ignore_update = False


class QTraitLineEdit(traitlets.HasTraits, QLineEdit, metaclass=MetaTraitQObject):
    value = traitlets.CUnicode()
    _value_signal = Signal(str)

    def __init__(self, parent, PySide2_QtWidgets_QWidget=None, NoneType=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__
        super().__init__(parent=parent)
        self.textChanged.connect(self.sp_value_changed)
        self._value_signal.connect(self.setText)
        self._ignore_update = False

    @traitlets.observe('value')
    def value_change(self, change):
        if not self._ignore_update:
            self._value_signal.emit(str(change['new']))

    def sp_value_changed(self, value):
        self._ignore_update = True
        self.value = value
        self._ignore_update = False


class QTraitLabel(traitlets.HasTraits, QLabel, metaclass=MetaTraitQObject):
    value = traitlets.CUnicode()

    def __init__(self, parent, PySide2_QtWidgets_QWidget=None, NoneType=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__
        super().__init__(parent=parent)

    @traitlets.observe('value')
    def value_change(self, change):
        self.setText(change['new'])


class QTraitSlider(traitlets.HasTraits, QSlider, metaclass=MetaTraitQObject):
    value = traitlets.CInt()
    _value_signal = Signal(int)

    def __init__(self, parent, PySide2_QtWidgets_QWidget=None, NoneType=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__
        super().__init__(parent=parent)
        self.valueChanged[int].connect(self.sp_value_changed)
        self._value_signal.connect(self.setValue)

        self._ignore_update = False


    @traitlets.observe('value')
    def value_change(self, change):
        if not self._ignore_update:
            self._value_signal.emit(change['new'])

    def sp_value_changed(self, value):
        self._ignore_update = True
        self.value = value
        self._ignore_update = False


class QTraitComboBox(traitlets.HasTraits, QComboBox, metaclass=MetaTraitQObject):
    index = traitlets.CInt()
    value = traitlets.CUnicode()

    _index_signal = Signal(int)
    _value_signal = Signal(str)
    _clear_signal = Signal()

    def __init__(self, parent, PySide2_QtWidgets_QWidget=None, NoneType=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__
        super().__init__(parent=parent)
        self.currentIndexChanged.connect(self.index_changed)
        self._index_signal.connect(self.setCurrentIndex)
        self._value_signal.connect(self.setCurrentText)
        self._clear_signal.connect(self.clear)

        self.currentIndexChanged.connect(self._index_changed)

        # For async reload purpose:
        self._load_items = []
        self._desired_index = 0

    @Slot(int)
    def _index_changed(self, index):
        if index == -1 and len(self._load_items) >0:
            self.addItems(self._load_items)
            self._load_items = []
            self.index=self._desired_index

    def clear_async(self):
        self._clear_signal.emit()

    @traitlets.observe('value')
    def value_changed(self, change):
        i = self.findText(change['new'])
        if i >= 0:
            self._index_signal.emit(i)

    @traitlets.observe('index')
    def index_change(self, change):
        self._index_signal.emit(change['new'])

    def index_changed(self, value):
        self.index = value
        self.value = self.currentText()

    def clear_and_load(self, new_items: list, try_keep_index=True):
        """ Clear list and load items.

        This function ensures clear() is called in the Gui thread, and then loads.

        @param new_items: List of items to load in combo box
        @param try_keep_index: If true, index is set to current matching item
        @return:
        """

        # Check if current item is in list:
        cur = self.value

        # Check if old item remains, so we can keep index:
        self._desired_index = 0
        if try_keep_index:
            if cur in new_items:
                for i, name in enumerate(new_items):
                    if name == cur:
                        self._desired_index = i
                        break

        # Clear list if items exist:
        if self.count() > 0:
            # We clear async:
            self._load_items = new_items  # Set items to load after currentIndex has been reset
            self.clear_async()            # Activate clear
        else:
            self.addItems(new_items)      # No old items, set new items


class QTraitProgressBar(traitlets.HasTraits, QProgressBar, metaclass=MetaTraitQObject):
    progress = traitlets.CFloat(min=0,max=1, default_value=0.0)
    active = traitlets.CBool(default_value=False)
    _progress_signal = Signal(int)

    def __init__(self, parent, PySide2_QtWidgets_QWidget=None, NoneType=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__
        super().__init__(parent=parent)
        self.setMinimum(0)
        self.setMaximum(100)
        self._progress_signal.connect(self.setValue)
        self.hide()

        self._trait_links = []
        self._linked_objects = {}

    @traitlets.observe('progress')
    def value_changed(self, change):
        self._progress_signal.emit(int(change['new']*100))

    @traitlets.observe('active')
    def active_changed(self, change):
        if change['new']:
            self.show()
        else:
            self.hide()

    def link(self, obj: IProcessStatus, keep_existing_links=False):
        # Clear existing links:
        obj.observe(self.value_changed, 'progress')
        obj.observe(self.active_changed, 'active')
        self.progress = obj.progress
        self.active = obj.active
        self._linked_objects[obj] = obj

    def unlink(self, prog_obj: IProcessStatus = None):
        if prog_obj in self._linked_objects:
            obj = self._linked_objects.pop(prog_obj)
            prog_obj.unobserve(self.value_changed, 'progress')
            prog_obj.unobserve(self.active_changed, 'active')


class QTraitCheckBox(traitlets.HasTraits, QCheckBox, metaclass=MetaTraitQObject):
    value = traitlets.CBool()
    _value_signal = Signal(Qt.CheckState)

    def __init__(self, parent, PySide2_QtWidgets_QWidget=None, NoneType=None, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__
        super().__init__(parent=parent)
        self.stateChanged.connect(self.value_changed)
        self._value_signal.connect(self.setCheckState)

    @traitlets.observe('value')
    def index_change(self, change):
        if change['new']:
            self._value_signal.emit(Qt.Checked)
        else:
            self._value_signal.emit(Qt.Unchecked)

    def value_changed(self, value):
        self.value = self.isChecked()


if __name__=="__main__":

    test = QComboBox()
    test.currentIndexChanged()
