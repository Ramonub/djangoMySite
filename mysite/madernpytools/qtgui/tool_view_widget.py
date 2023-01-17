import traitlets, os, re
import madernpytools.backbone as mbb
from PySide2.QtWidgets import QLabel, QFrame
from PySide2.QtGui import Qt
from madernpytools.qtgui.ui_tool_view_widget import Ui_ToolViewWidget
from madernpytools.qtgui.traitlet_widgets import AbstractWidget, MetaTraitQObject
from madernpytools.tools.utilities import ListofDict
from madernpytools.backbone import HasTraitLinks


class ToolViewFactory(mbb.IClassFactory):
    """ Factory class which allows to generate class instances from this module

    """

    @staticmethod
    def get(name):
        return eval(name)


class QTraitSignalLabel(mbb.TraitsXMLSerializer, QLabel, metaclass=MetaTraitQObject):
    value = traitlets.CFloat(default_value=0.0)
    name = traitlets.CUnicode(default_value='')
    unit = traitlets.CUnicode(default_value='')
    x = traitlets.CFloat(default_value=0.0)
    y = traitlets.CFloat(default_value=0.0)

    def __init__(self, parent=None, x=0.0, y=0.0, name='default', unit='-'):
        """

        @param parent: Parent of Signal label
        @param x: relative location in x-direction
        @param y: relative location in y-direction
        @param name: task_name of signal
        @param unit: unit of signal
        """

        names = ['name', 'unit', 'x', 'y']
        super().__init__(parent=parent,
                         var_names_mapping=list(zip(names, names)))

        self.unit = unit
        self.name = name
        self.x = x
        self.y = y
        self.setAlignment(Qt.AlignCenter)
        self.setMargin(3)
        self.setFrameShadow(self.Sunken)
        self.setFrameShape(self.Panel)
        self.setToolTip(f'Signal name: {self.name}')
        self.setAutoFillBackground(True)

        if self.parent():
            self.update_location()
        self._value_change(None)

    @traitlets.observe('value')
    def _value_change(self, change):
        self.setText(f'{self.value:>10_.3f} ({self.unit})')
        self.resize(self.sizeHint())

    def update_location(self):
        """

        @return:
        """
        # Get size indication
        if self.parent() is not None:
            size = self.parent().size()
            w = size.width()
            h = size.height()
            self.move(int(w * self.x), int(h * self.y))


class ToolViewWidget(AbstractWidget, Ui_ToolViewWidget, mbb.TraitsXMLSerializer):
    input_data = ListofDict()
    image_url = traitlets.CUnicode(default_value=':/newPrefix/ema_schematic_tooling.png')
    _signal_labels = traitlets.Dict(default_value=dict())

    def __init__(self, parent=None, image_url='', signal_labels=None):
        if image_url == '':
            image_url = ':/newPrefix/ema_schematic_tooling.png'
        if signal_labels is None:
            signal_labels = {}

        super().__init__(parent=parent, image_url=image_url,
                         var_names_mapping=[('signal_labels', '_signal_labels'), ('image_url', 'image_url')])

        # Set labels only after object is fully initialized:
        self._signal_labels = signal_labels

    @traitlets.observe('image_url')
    def image_change(self, change):
        self.image_widget.setStyleSheet(u"image: url({});".format(change.new))

    @traitlets.observe('input_data')
    def _input_change(self, change):
        for key in change.new.get_keys():
            if key in self._signal_labels:
                self._signal_labels[key].value = change.new[-1][key]

    @traitlets.observe('_signal_labels')
    def _signallabel_change(self, change):
        # Remove old:
        if isinstance(change.old, dict):
            for key, item in change.old.items():
                if isinstance(item, QTraitSignalLabel):
                    item.deleteLater()

        # Add new
        for key, item in change.new.items():
            if isinstance(item, QTraitSignalLabel):
                item.setParent(self)
                item.update_location()

    def add_label(self, item: QTraitSignalLabel):
        item.setParent(self)
        item.update_location()
        self._signal_labels[item.name] = item

    @property
    def signal_labels(self) -> dict:
        return self._signal_labels

    def setParent(self, parent):
        AbstractWidget.setParent(self, parent)

        # Update children
        for key, item in self._signal_labels.items():
            item.update_location()


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QSizePolicy

    # Define MainWindow & Layout:
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    hlayout = QHBoxLayout(mainWindow.centralWidget())

    # Define test widget:
    wid = ToolViewWidget()
    #hlayout.addWidget(wid)

    wid.add_label(QTraitSignalLabel(x=0.82, y=0, name='OS_acc', unit='N'))
    wid.add_label(QTraitSignalLabel(x=0.02, y=0, name='DS_acc', unit='N'))
    wid.add_label(QTraitSignalLabel(x=0.20, y=0.38, name='os_upper', unit='N'))
    wid.add_label(QTraitSignalLabel(x=0.20, y=0.5, name='os_lower', unit='N'))
    wid.add_label(QTraitSignalLabel(x=0.65, y=0.38, name='ds_upper', unit='N'))
    wid.add_label(QTraitSignalLabel(x=0.65, y=0.5, name='ds_lower', unit='N'))

    mbb.ET.ElementTree(wid.to_xml()).write('./toolset_view.xml')

    test = ToolViewWidget.from_xml(wid.to_xml(), class_factory=ToolViewFactory())
    test.setParent(mainWindow)
    hlayout.addWidget(test)


    # Show window
    mainWindow.resize(wid.size())
    mainWindow.show()
    sys.exit(app.exec_())


