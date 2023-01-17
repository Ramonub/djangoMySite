import sys, traitlets, logging, asyncio, os
from PySide2.QtWidgets import QLabel,  QApplication
import numpy as np
from madernpytools.qtgui.traitlet_widgets import AbstractMainWindow
from madernpytools.qtgui.ui_data_gui import Ui_DataGui

from qasync import QEventLoop

from madernpytools.daq_datalogging import NIDataLogger, get_default_nilogger
import madernpytools.filtering as mfilt
from madernpytools.filtering import SignalKeyList, FilterList, AbstractDataProcessor, MathOperation, NumpyOperation
import madernpytools.data_processing.DAQ as mdaq
import madernpytools.log as mlog
import madernpytools.signal_handling as msig
import madernpytools.daq_datalogging as mlogging
import madernpytools.tools.utilities as mutils
import madernpytools.backbone as mbb
from madernpytools.qtgui.tool_view_widget import QTraitSignalLabel
from madernpytools.qtgui.autosave_log_widget import logger as wlog_logger

import scipy as sp
import scipy.signal

from madernpytools.signal_handling import ISignalKeyList

mod_dir = os.path.dirname(__file__)

logger = logging.getLogger(f'madernpytools.{__name__}')


class ClassFactory(mbb.IClassFactory):
    """ Factory class which allows to generate class instances from this module

    """

    @staticmethod
    def get(name):
        if name in sys.modules[__name__].__dict__:
            return eval(name)
        else:
            for mod in [mdaq, mlogging, mfilt]:
                if name in getattr(mod, '__dict__'):
                    return getattr(mod, 'ClassFactory').get(name)
            raise TypeError('Could not find {}'.format(name))


class CSDSettings(mbb.TraitsXMLSerializer):
    window = traitlets.CUnicode(default_value='hann')
    scaling = traitlets.CUnicode(default_value='spectrum')
    nperseg = traitlets.CInt(default_value=10)
    noverlap = traitlets.CInt(default_value=5)
    fs = traitlets.CInt(default_value=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_csd_dict(self):
        return {'window': self.window, 'nperseg': self.nperseg,
                'noverlap': self.noverlap, 'fs': self.fs, 'scaling': self.scaling}


class SpectralAnalyzer(mfilt.AbstractFilter):
    # Filter settings:
    filter_settings = CSDSettings()
    frequency_output =mutils.ListofDict()
    signal_keys = SignalKeyList(items=[])

    def __init__(self):
        super().__init__()
        asyncio.create_task(self._update_loop())

    @traitlets.default('frequency_output')
    def _default_foutput(self):
        return mutils.ListofDict()

    async def _update_loop(self):
        """ Loop that continuously updates the

        @return:
        """

        while True:
            data = self.input_data[:]
            available_keys = data.get_keys()
            if len(data) > self.filter_settings.nperseg:
                output = {}
                for i, key in enumerate(self.signal_keys):
                    if not key in available_keys:
                        continue

                    key_data = data.get_key_values(key=key)
                    freq, Sxx = await asyncio.to_thread(sp.signal.csd, key_data, key_data, **self.filter_settings.get_csd_dict())
                    output[key] = Sxx
                    if i == 0:
                        output['frequencies'] = freq
                    await asyncio.sleep(0.1) # Pause computation to allow the remainder of the program to perform actions

                try:
                    if len(output) > 0:
                        self.frequency_output = mutils.ListofDict([output])
                    else:
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(e)
            else:
                await asyncio.sleep(0.1)

    async def apply_filter(self):
        pass

    @property
    def required_input_keys(self) -> ISignalKeyList:
        return SignalKeyList([])

    @property
    def added_keys(self) -> ISignalKeyList:
        return SignalKeyList(list(self.frequency_output.keys()))


class OnlineDataProcessor(AbstractDataProcessor):

    def __init__(self, filter_list: FilterList =None, sampling_rate=10, *args, **kwargs):

        if filter_list is None:
            filter_list = FilterList(sampling_rate=sampling_rate, input_keys=self.required_input_keys)

            filter_list.append(mfilt.ZeroSignals())
            operations = MathOperation(filter_operations=[NumpyOperation(name='os_gap', function=np.add,
                                                                        argument_keys=['os_upper', 'os_lower']),
                                                          NumpyOperation(name='ds_gap', function=np.add,
                                                                        argument_keys=['ds_upper', 'ds_lower'])
                                                          ]
                                       )

            filter_list.append(operations)

        super().__init__(filter_list=filter_list, sampling_rate=sampling_rate)

        self._zero_signals = None
        for item in filter_list:
            if isinstance(item, mfilt.ZeroSignals):
                self._zero_signals = item

    async def apply_filter(self):
        self.filter_list.input_data = self.input_data[:]

    @property
    def zero_signals(self) -> mfilt.ZeroSignals:
        return self._zero_signals

    @traitlets.observe('filter_list')
    def _filter_list_change(self, change):

        # Unlink existing links:
        for key, l in self._filter_links.items():
            if isinstance(l, traitlets.link):
                l.unlink()
        self._filter_links = {}

        self._process_merger.unobserve_process(change.old)
        self._process_merger.observe_process(change.new)

        # Link in and outputs:
        # Data flow:
        self._filter_links['filt->sel'] = traitlets.link((self.filter_list, 'output_data'),
                                                         (self, 'output_data'))

        # Sync sampling rate:
        self._filter_links['sampling_rate'] = (traitlets.link((self, 'sampling_rate'),
                                                              (self.filter_list, 'sampling_rate')))

    @property
    def required_input_keys(self):
        return self.filter_list.required_input_keys


class DataGui(AbstractMainWindow, Ui_DataGui, mbb.IXML, mbb.HasTraitLinks):
    data_logger = NIDataLogger()
    data_processor = OnlineDataProcessor()
    buffer = traitlets.TraitType()

    def __init__(self,
                 data_logger: NIDataLogger = None,
                 data_processor: OnlineDataProcessor = None,
                 buffer_size=-1,
                 signal_labels: dict = None):

        super().__init__()

        self.btn_zero.clicked.connect(self._zero_sensors)

        if signal_labels is not None:
            for key, label in signal_labels.items():
                self.tool_view.add_label(label)

        self.actionConnect.triggered.connect(self._connect)

        self._spectral_analyzer = SpectralAnalyzer()

        # Link logger task data to data processor and tool display::
        self.data_logger.task.observe(self._datalogger_change, 'output_data')
        self.task_links = [
                           traitlets.link((self.data_logger.task, 'output_data'), (self.data_processor, 'input_data')),
                           ]

        self.buffer_links = [
                           traitlets.link((self.buffer, 'output_data'), (self.time_display, 'input_data')),
                           traitlets.link((self.buffer, 'output_data'), (self._spectral_analyzer, 'input_data')),
                           ]

        # Link data processor signal selection to buffer & displays:
        self.dataprocessor_links = [traitlets.link((self.data_processor, 'output_data'), (self.buffer, 'input_data')),
                                    traitlets.link((self.data_processor, 'output_data'), (self.tool_view, 'input_data'))
                                    ]

        # Link selected items to spectral analyzer selection::
        traitlets.link((self.frequency_display.signal_selection, 'selected_items'),
                       (self._spectral_analyzer, 'signal_keys'))

        # Link spectral analyzer to spectral display
        traitlets.link((self._spectral_analyzer, 'frequency_output'), (self.frequency_display, 'input_data'))
        self.frequency_display.x_key = 'frequencies'

        # Data logger:
        self.log_control.log = self.data_logger.log
        self.data_logger.observe(self._connection_changed, 'connected')

        # Set Logger, DataProcessor and Buffer:
        if isinstance(data_logger, NIDataLogger):
            self.data_logger = data_logger

        if isinstance(data_processor, OnlineDataProcessor):
            self.data_processor = data_processor

        if buffer_size > -1:
            self.buffer = msig.Buffer(n=buffer_size, signal_keys=self.data_logger.task.added_keys)
        else:
            self.buffer = msig.Buffer(n=int(self.data_logger.task.sampling_rate),
                                      signal_keys=self.data_logger.task.added_keys)

        # Status bar:
        self._status_label = QLabel(self.statusbar)
        self.statusbar.addPermanentWidget(self._status_label)

        logger.info(f'Time display id: {id(self.time_display)}')

    @traitlets.default('data_processor')
    def _default_dataprocessor(self):
        dp = OnlineDataProcessor()
        return dp

    @traitlets.observe('data_processor')
    def _dataprocessor_change(self, change):
        # Unlink
        for i, l in enumerate(self.task_links):
            if isinstance(l.target[0], type(change.new)):
                l.unlink()

        logger.info('---- Updated data_processor links ---')
        self.update_links(src_obj=change.new, list_attribute_name='dataprocessor_links')
        for l in self.dataprocessor_links:
            logger.info(f'source {type(l.source[0]).__name__}.{l.source[1]} ({id(l.source[0])})'
                        f', target {type(l.target[0]).__name__}.{l.target[1]} ({id(l.target[0])}).')

        for i, l in enumerate(self.task_links):
            if isinstance(l.target[0], type(change.new)):
                self.task_links[i] = traitlets.link(l.source, (change.new, l.target[1]))
                logger.info(f'source {type(l.source[0]).__name__}.{l.source[1]} ({id(l.source[0])})'
                            f', target {type(l.target[0]).__name__}.{l.target[1]} ({id(l.target[0])}).')

        self._update_available_keys()
        self.data_processor.zero_signals.signal_keys = self.data_logger.task.added_keys

    @traitlets.default('buffer')
    def _default_buffer(self):
        return msig.Buffer(n=1000, signal_keys=SignalKeyList())

    @traitlets.observe('buffer')
    def _buffer_change(self, change):

        try:
            change.old.unobserve(self._buffer_output, 'output_data')
        except:
            pass
        finally:
            change.new.observe(self._buffer_output, 'output_data')

        self.update_links(src_obj=change.new, list_attribute_name='buffer_links')

        logger.info(f"Linking items to buffer id {id(change.new)}")
        for i, l in enumerate(self.dataprocessor_links):
            if isinstance(l.target[0], type(change.new)):
                l.unlink()
                self.task_links[i] = traitlets.link(l.source, (change.new, l.target[1]))
                l = self.task_links[i]
                logger.info(f'source {type(l.source[0]).__name__}.{l.source[1]} ({id(l.source[0])})'
                            f', target {type(l.target[0]).__name__}.{l.target[1]} ({id(l.target[0])})')
        logger.info(f"-------")

    def _buffer_output(self, change):
        pass
        #self.time_display.input_data = change.new
        #self._spectral_analyzer.input_data = change.new
        #logger.info(f'Buffer output change from {id(change.owner)}')

    @traitlets.default('data_logger')
    def _default_datalogger(self):
        return NIDataLogger()

    @traitlets.observe('data_logger')
    def _datalogger_change(self, change):
        if isinstance(change.new, NIDataLogger):
            # Manage observers:
            if isinstance(change.old, NIDataLogger):
                try:
                    change.old.unobserve(self._connection_changed, 'connected')
                    change.old.task.unobserve(self._datalogger_change, 'output_data')
                except Exception as exp:
                    logger.error(exp)
                finally:
                    change.new.observe(self._connection_changed, 'connected')
                    change.new.task.observe(self._datalogger_change, 'output_data')

            # Allow log-control through GUI:
            self.log_control.log = self.data_logger.log

            # Link management
            self.update_links(src_obj=change.new.log, list_attribute_name='logger_links')
            self.update_links(src_obj=change.new.task, list_attribute_name='task_links')

            # Update spectral analyzer:
            self._spectral_analyzer.filter_settings.fs = int(change.new.task.sampling_rate)
            self._spectral_analyzer.filter_settings.nperseg = int(change.new.task.sampling_rate)/2
            self._spectral_analyzer.filter_settings.noverlap = int(change.new.task.sampling_rate)/4

            self._update_available_keys()

    def _connect(self):
        self.data_logger.connect()

    def _connection_changed(self, change):
        if change.new:
            self._status_label.setText('Connected')
        else:
            self._status_label.setText('Not connected')

    def closeEvent(self, event) -> None:
        self.data_logger.disconnect()

    def _update_available_keys(self):
        new_list = self.data_processor.added_keys + self.data_logger.task.added_keys
        new_list.remove('time')
        self.time_display.signal_selection.available_items = new_list
        self.frequency_display.signal_selection.available_items = new_list

    def _zero_sensors(self):
        self.data_processor.zero_signals.zero_signals()

    def to_xml(self):
        """

        @return:
        """

        # Create Root:
        xml_root = mbb.ET.Element(type(self).__name__)
        xml_root.set('Type', type(self).__name__)

        # Serialize:
        mbb.XMLSerializer.serialize_into_root(self.data_logger, xml_root, tag='data_logger')
        mbb.XMLSerializer.serialize_into_root(self.data_processor, xml_root, tag='data_processor')
        mbb.XMLSerializer.serialize_into_root(self.buffer.buffer_size, xml_root, tag='buffer_size')
        mbb.XMLSerializer.serialize_into_root(self.tool_view.signal_labels, xml_root, tag='signal_labels')

        return xml_root

    @staticmethod
    def from_xml(root, *args, **kwargs):

        item_dict = {}
        for child in root:
            item_dict[child.tag] = mbb.XMLSerializer.deserialize_xml_item(child, ClassFactory())

        return DataGui(**item_dict)


def get_gap_gui():

    # Get task
    ni_logger = get_default_nilogger()

    # Define data GUI
    data_gui = DataGui(data_logger=ni_logger, buffer_size=ni_logger.task.sampling_rate*10)

    # Set tool display:
    wid = data_gui.tool_view
    # Set image:
    wid.image_url = ':/newPrefix/ema_schematic_tooling.png'
    # Set labels:
    wid.add_label(QTraitSignalLabel(x=0.02, y=0, name='os_acc', unit='m/s2'))
    wid.add_label(QTraitSignalLabel(x=0.82, y=0, name='ds_acc', unit='m/s2'))
    wid.add_label(QTraitSignalLabel(x=0.23, y=.43, name='os_gap', unit='mu'))
    wid.add_label(QTraitSignalLabel(x=0.57, y=.43, name='ds_gap', unit='mu'))
    wid.add_label(QTraitSignalLabel(x=0.20, y=0.32, name='os_upper', unit='mu'))
    wid.add_label(QTraitSignalLabel(x=0.20, y=0.54, name='os_lower', unit='mu'))
    wid.add_label(QTraitSignalLabel(x=0.60, y=0.32, name='ds_upper', unit='mu'))
    wid.add_label(QTraitSignalLabel(x=0.60, y=0.55, name='ds_lower', unit='mu'))
    wid.add_label(QTraitSignalLabel(x=0.4, y=0.70, name='ToolAngle', unit='rad'))

    return data_gui


def main(settings_file: str=''):
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    filename = settings_file if settings_file !='' else f'{mod_dir}/../settings/data_gui.xml'
    if not os.path.exists(filename):
        data_gui = get_gap_gui()

        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        mutils.XMLFileManager.save(data_gui, filename)
    else:
        data_gui = mutils.XMLFileManager.load(filename, class_factory=ClassFactory())

    data_gui.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, filename='example_log.log')
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s %(threadName)s"))
    logger.addHandler(stream)
    #msigviewer.logger.addHandler(stream)
    #msig.logger.addHandler(stream)
    #mdaq.logger.addHandler(stream)
    mlog.logger.addHandler(stream)
    wlog_logger.addHandler(stream)
    #main()
    main('./settings/vibr_meas_eindmontage_sim.xml')


