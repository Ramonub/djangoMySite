from madernpytools.backbone import DataPublisher, DataSubscriber, AbstractDataProcessor, EventPublisher, SimpleEventSubscriber
from madernpytools.backbone import Timeout, EventSubscriber, ArgumentVerifier, IXML
import madernpytools.backbone as mbb
from madernpytools.tools.utilities import ListofDict
import threading as thrd
import time, os, traitlets, logging
import numpy as np

import xml.etree.cElementTree as ET

logger = logging.getLogger(f'madernpytools.{__name__}')


class ISignalKeyList(traitlets.TraitType):

    def __add__(self, other: object):
        raise NotImplementedError()

    def __sub__(self, other: object):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __contains__(self, item:(str, tuple)):
        raise NotImplementedError()

    def index(self, item:(str, tuple)):
        raise NotImplementedError()

    def remove(self, item:(str, tuple)):
        raise NotImplementedError()

    def append(self, item:(str, tuple)):
        raise NotImplementedError()

    def insert(self, index: int, item:(str, tuple)):
        raise NotImplementedError()

    def sort(self, *args, **kwargs):
        raise NotImplementedError()


class ISignalProvider(object):

    @property
    def required_input_keys(self) -> ISignalKeyList:
        """ The keys required to perform the filter action. These keys should be available in the provided 'input_data'
        ListOfDict(). If keys are not available, an error is thrown

        :return:
        """
        raise NotImplementedError()

    @property
    def added_keys(self) -> ISignalKeyList:
        """ The keys which are added by the filter.
        :return:
        """
        raise NotImplementedError()


class SignalKeyList(ISignalKeyList, mbb.XMLSerializer):

    def __init__(self, items: list=None, **kwargs):
        """ List of signal keys. Contains either strings or tuple of strings e.g.:
        [('x', 'x_raw'), 'height', 'width']

        SignalKeyList supports addition and substraction, thereby considering:
        * Options which are defined by tuples ('x', 'x_raw') indicating 'x' or 'x_raw'), and;
        * Not operators '!', which cause removal of the signal i.e. ['x', 'y'] + ['!y'] = ['x']
        * Not any operators '!*', which cause removal of the signal i.e. ['x', 'y'] + ['!*'] = []

        :param items:
        """

        if items is None:
            items = []
        if not isinstance(items, list):
            raise TypeError('Expected list, received {}'.format(type(items)))

        mbb.XMLSerializer.__init__(self, var_names_mapping=[('items', '_list')])
        ISignalKeyList.__init__(self, **kwargs)

        if items is None:
            items = []
        self._list = items

    def append(self, item: (str, tuple)):
        """ Add item to signal key list

        :param item:
        :return:
        """
        if isinstance(item, (str, tuple)) and (item not in self._list):
            self._list.append(item)
        elif not isinstance(item, (str, tuple)):
            raise TypeError('Expected str but received {}'.format(type(item)))

    def index(self, item, start: int=0, end: int=-1):
        """

        :param element:
        :param start:
        :param end:
        :return:
        """

        if isinstance(item, (tuple, str)):
            return self._list.index(item)
        else:
            raise TypeError('Type should be tuple or string, not {}'.format(type(item)))

    def insert(self, index: int, item: str):
        if isinstance(item, str) and (item not in self):
            # Not in list, and received str
            self._list.insert(index, item)
        elif isinstance(item, str):
            # Inappropriate type:
            raise TypeError('Expected str but received {}'.format(type(item)))

    def remove(self, item: str):
        self._list.remove(item)

    def __add__(self, other: ISignalKeyList):
        """ Add keys to list

        :param other:
        :return:
        """
        new_list = SignalKeyList(self._list[:])
        for i, key in enumerate(other):
            if '!*' == key:
                return other[(i+1):]
            elif key[0] == '!':
                # Remove key, because it is preceeded with !
                if key[1:] in new_list:
                    new_list.remove(key[1:])
            else:
                if not (key in new_list):
                    new_list.append(key)

        return new_list

    def __sub__(self, other: ISignalKeyList):
        """ Remove keys from list

        :param other:
        :return:
        """
        new_list = SignalKeyList(self._list[:])
        for key in other:
            if '!*' == key:
                continue
            elif key[0] == '!':
                if not (key[1:] in new_list):
                    new_list.append(key[1:])
            else:
                if key[:] in new_list:
                    new_list.remove(key)
        return new_list

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def __contains__(self, item: (str, tuple)):
        """ Check if SignalKeyList contains item

        :param item: Either a str or a tuple of string. Tuple acts as an 'or'
        :return:
        """

        if isinstance(item, tuple):
            return np.logical_or.reduce([sub_item in self for sub_item in item])
        elif isinstance(item, str):
            item_found = False

            for key in self._list:
                if isinstance(key, tuple):
                    item_found = np.logical_or.reduce([sub_key==item for sub_key in key])
                elif isinstance(key, str):
                    item_found = key == item

                if item_found:
                    break
            return item_found
        else:
            raise TypeError('Type should be tuple or string, not {}'.format(type(item)))

    def __str__(self):
        return str(self._list)

    def __eq__(self, other):
        return (len(self) == len(other)) and all([key in self for key in other])

    def sort(self, key=None, reverse=False):
        self._list.sort(key=key, reverse=reverse)
        raise NotImplementedError()


class AbstractSignal(mbb.TraitsXMLSerializer):
    """Interface class for signals"""
    name = traitlets.CUnicode(default_value='')
    unit = traitlets.CUnicode(default_value='')
    module = traitlets.CUnicode(default_value='')
    channel_index = traitlets.CInt(default_value=-1)
    data_index = traitlets.CInt(default_value=-1)
    scale = traitlets.CFloat(default_value=1.0)

    def __init__(self, name='', unit='', module='', channel_index=-1, scale=1.0, data_index=-1):
        super().__init__(name=name,
                         unit=unit,
                         module=module,
                         channel_index=channel_index,
                         data_index=data_index,
                         scale=scale)


class SummedSignal(AbstractSignal, DataPublisher):

    def __init__(self, signal1: AbstractSignal, signal2: AbstractSignal, name, unit):
        AbstractSignal.__init__(self)

        self._data = None
        self._calib_value = 0.0

        self._signal1 = signal1
        self._signal2 = signal2

        self._name = name
        self._unit = unit

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def data_index(self):
        return self._signal1.data_index, self._signal2.data_index

    @property
    def module(self):
        return self._signal1.module, self._signal2.module

    @property
    def channel_index(self):
        return self._signal1.channel_index, self._signal2.channel_index

    @property
    def data(self):
        return self._data - self._calib_value

    @data.setter
    def data(self, value):
        """ Set data to this sensor
        :param value: N-dimensional array of data
        :return:
        """
        self._data = value.sum(axis=1)

        # Notify subscribers
        if len(self._subscribers) > 0:
            self.raise_event(self.data)

    @property
    def value(self):
        return self._signal1.value + self._signal2.value

    @property
    def scale(self):
        return self._signal1.scale, self._signal2.scale

    def cb_new_data(self, publisher):
        pass

    @staticmethod
    def from_xml(xml_element):
        pass

    def to_xml(self):
        pass

    def set_data_index(self, index):
        pass

    def set_zero(self):
        """
        Take current measurement as zero

        :return:
        """
        if self._data is None:
            self._calib_value=0.0
        else:
            n_vals = min(self._data.shape[0], 10)
            self._calib_value = self._data[-n_vals:-1, ].mean()

    @property
    def calibration_value(self):
        return -1


class Signal(AbstractSignal, DataPublisher):

    def __init__(self, name, unit, data_index=None, module='', channel_index=-1, scale=1.0):
        AbstractSignal.__init__(self)
        DataPublisher.__init__(self)

        # Signal Info:
        self._data = None
        self._name = ArgumentVerifier(str, None) .verify(name)
        self._unit = ArgumentVerifier(str, None).verify(unit)
        self._module = ArgumentVerifier(str, None).verify(module)
        self._channel_index = ArgumentVerifier(int, None).verify(channel_index)
        self._data_index = ArgumentVerifier((int, list), -1).verify(data_index)
        self._calib_value = 0.0
        self._scale = scale

    @property
    def calibration_value(self):
        """Signal calibration value. This value represents the offset between the signal true value and
         its calibrated value"""
        return self._calib_value

    @property
    def name(self):
        """
        Signal task_name
        @return:
        """
        return self._name

    @property
    def scale(self):
        """Signal scaling

        @return:
        """
        return self._scale

    @property
    def unit(self):
        """Signal Unit"""
        return self._unit

    @property
    def module(self):
        """Module task_name to which this signal is assigned

        @return:
        """
        return self._module

    @property
    def channel_index(self):
        """Module Channel on which the signal is measured

        @return:
        """
        return self._channel_index

    @property
    def data_index(self):
        """Signal index in the task data stream

        @return:
        """
        return self._data_index

    @property
    def data(self):
        """ Signal data

        @return:
        """
        if self._data is None:
            return None
        else:
            return self._data - self._calib_value

    @data.setter
    def data(self, value):
        """ Set data to this sensor
        :param value: N-dimensional array of data
        :return:
        """
        self._data = value*self._scale

        # Notify subscribers
        if len(self._subscribers) > 0:
            self.raise_event(self.data)

    @property
    def value(self):
        """Current signal value

        @return:
        """
        value = None
        if type(self.data) is np.ndarray:
            if self._data.ndim == 1:
                value = self.data[-1]
            elif self.data.ndim == 2:
                value = self.data[-1, :]
        return value

    def cb_new_data(self, publisher):
        """Update signal value
        :param publisher : publisher that contains the update"""
        # Update value
        self.data = publisher.get_data()[:, self._data_index]

    def set_zero(self):
        """
        Take current measurement as zero

        :return:
        """
        if self._data is None:
            self._calib_value=0.0
        else:
            n_vals = min(self._data.shape[0], 10)
            self._calib_value = self._data[-n_vals:-1, ].mean()

    @staticmethod
    def from_xml(xml_element):
        """ Create signal from xml_settings

        :param xml_element:
        :return:
        """
        return Signal(name=xml_element.find('task_name').text,
                      data_index=int(xml_element.find('data_index').text),
                      module=xml_element.find('device_name').text,
                      channel_index=int(xml_element.find('channel').text)
                      )

    def to_xml(self):
        """ Write signal settings to XML
        :return:
        """
        signal_settings = ET.Element('signal')
        signal_settings.set('task_name', self.name)

        # Values:
        name = ET.SubElement(signal_settings, 'task_name')
        name.text = s.name

        module_name = ET.SubElement(signal_settings, 'device_name')
        module_name.text = self.module

        channel = ET.SubElement(signal_settings, 'channel')
        channel.text = str(self.channel_index)

        data_index = ET.SubElement(signal_settings, 'data_index')
        data_index.text = str(self.data_index)

        unit = ET.SubElement(signal_settings, 'unit')
        unit.text = self.unit

        return signal_settings

    def set_data_index(self, index):
        """Set index of signal in task data stream

        @param index:
        @return:
        """
        self._data_index = index


class SignalDict(dict, AbstractSignal):

    def __init__(self, *args):
        """Dictionary of signals"""
        super().__init__(*args)
        #dict.__init__(self, *args)
        #AbstractSignal.__init__(self)

    def to_xml(self):
        """Convert instance to XML"""

        # Create root:
        xml_root = ET.Element(type(self).__name__)
        xml_root.set('Type', type(self).__name__)

        # Serialize all _sensors in dict:
        for key, item in self.items():
            # Set XML data type:
            xml_item = ET.SubElement(xml_root, key)
            xml_item.set('Type', type(item).__name__)

            # Add data to XML entry:
            if isinstance(item, IXML):
                xml_item.append(item.to_xml())
            elif isinstance(item, (str, float, int)):
                xml_item.text = str(item)
            else:
                raise ValueError("Cannot convert {0} into XML format".format(item))

        return xml_root

    @staticmethod
    def from_xml(xml_element, class_factory: mbb.IClassFactory = None):
        """Initialize instance from XML data"""

        # Define class factory:
        class_factory = class_factory if class_factory is not None else mbb.IClassFactory

        item_dict = {}
        for child in xml_element:
            # Identify type:
            child_type = class_factory.get(child.get('Type'))
            key = child.tag

            # Extract dict:
            if issubclass(child_type, IXML):
                item_dict[key] = child_type.from_xml(child.find(child.get('Type')), class_factory)
            else:
                item_dict[key] = child_type(child.text)

        # Construct object:
        item_type = class_factory.get(xml_element.get('Type'))

        return item_type(**item_dict)

    def __setitem__(self, key, item):
        """Set signal with key"""
        if isinstance(item, AbstractSignal):
            dict.__setitem__(self, key, item)
        else:
            raise ValueError("Item is not of type {0}".format(
                AbstractSignal.__name__))

    def get_signal_pair(self, x_key, y_key):
        """Get pair of signals"""
        ind = [self[x_key].data_index, self[y_key].data_index]
        new_pair = Signal(name='{0},{1}'.format(x_key, y_key), unit=[x_key, y_key], data_index=ind)

        return new_pair

    def _get_attr(self, attr):
        """Get attributed"""
        items = {}
        for key, item in self.items():
            items[key] = getattr(item,attr)

        return items

    @property
    def name(self):
        """Dict of signal names"""
        return self._get_attr('task_name')

    @property
    def calibration_value(self):
        return self._get_attr('calibration_value')

    @property
    def scale(self):
        return self._get_attr('scale')

    @property
    def unit(self):
        """Dict of signal units"""
        return self._get_attr('unit')

    @property
    def module(self):
        """Dict of signal modules"""
        return self._get_attr('module')

    @property
    def channel_index(self):
        """Dict of signal modules"""
        return self._get_attr('channel_index')

    @property
    def data_index(self):
        """Dict of signal modules"""
        return self._get_attr('data_index')

    @property
    def data(self):
        """Dict of signal modules"""
        return self._get_attr('data')

    @data.setter
    def data(self, data):
        """ Set data to this sensor
        :param data: N-dimensional array of data
        :return:
        """
        raise NotImplementedError()

    @property
    def value(self):
        return self._get_attr('value')

    def cb_new_data(self, publisher):
        """Update signal value
        :param publisher : publisher that contains the update"""

        data = publisher.get_data()

        # Update data of signals:
        for key, item in self.items():
            item.data = data[:, item.data_index]

        # Publish data:
        self.raise_event(self)

    def set_zero(self):
        """
        Take current measurement as zero
        :return:
        """
        for key, item in self.items():
            item.set_zero()

    @property
    def signal_names(self):
        """List of signal names sorted by data index"""
        return [key for key, _ in sorted(self.items(), key=lambda item: item[1].data_index)]


class AbstractSignalSelector(DataSubscriber):

    def __init__(self):
        DataSubscriber.__init__(self)

    def cb_new_data(self, publisher):
        raise NotImplementedError()


class SignalList(list, AbstractSignal):

    def __init__(self, *args):
        list.__init__(self, *args)
        AbstractSignal.__init__(self)

    def append(self, signal):
        if isinstance(signal, AbstractSignal):
            list.append(self, signal)
        else:
            raise ValueError('signal should be derived from AbstractSignal')

    def __setitem__(self, index, signal):
        if isinstance(signal, AbstractSignal):
            list.__setitem__(self, index, signal)
        else:
            raise ValueError("Item is not of type {0}".format(
                AbstractSignal.__name__))

    def get_by_name(self, name):
        """Get signal by task_name"""
        found = None
        for s in self:
            if s.task_name == name:
                found = s
                break

        return found

    def _get_attr(self, attr):
        """Get attributed"""
        items = []
        for item in self:
            items.append(getattr(item, attr))

        return items

    @property
    def name(self):
        """Dict of signal names"""
        return self._get_attr('task_name')

    @property
    def calibration_value(self):
        return self._get_attr('calibration_value')

    @property
    def unit(self):
        """Dict of signal units"""
        return self._get_attr('unit')

    @property
    def module(self):
        """Dict of signal modules"""
        return self._get_attr('module')

    @property
    def channel_index(self):
        """Dict of signal modules"""
        return self._get_attr('channel_index')

    @property
    def data_index(self):
        """Dict of signal modules"""
        return self._get_attr('data_index')

    @property
    def data(self):
        """Dict of signal modules"""
        return self._get_attr('data')

    @data.setter
    def data(self, data):
        """ Set data to this sensor
        :param data: N-dimensional array of data
        :return:
        """
        raise NotImplementedError()

    @property
    def value(self):
        return self._get_attr('value')

    def cb_new_data(self, publisher):
        """Update signal value
        :param publisher : publisher that contains the update"""

        data = publisher.get_data()

        # Update data of signals:
        for item in self:
            item.data = data[:, item.data_index]

        # Publish data:
        self.raise_event(self)

    def set_zero(self):
        """
        Take current measurement as zero
        :return:
        """
        for item in self:
            item.set_zero()


class SignalSelector(DataSubscriber):

    def __init__(self, data_step=1):
        """Selects groups of signals---e.g. (time, acc1), (time, acc2) from a data stream, and forwards them as a
        SignalList to subscribers.

        :param data_step: indicates the amount of data to stream, i.e. data[::data_step, :]. For example, 1 forwards
        all data, 2 forwards every other sample,
        """
        DataSubscriber.__init__(self)

        self._signals = SignalList()
        self.data_step = data_step
        self._event_new_data = DataPublisher()      # Internal event
        self._event_new_data.connect(self._signals) # Connect signals to internal event

    def add_selection(self, data_indices, name='', unit='', module='', channel_index=-1):
        signal = Signal(name, unit, data_index=data_indices, module=module, channel_index=channel_index)
        self._signals.append(signal) # add to object

    @property
    def signals(self):
        """List of signal selections"""
        return self._signals

    def cb_new_data(self, publisher):
        # Update internal data:
        self._event_new_data.raise_event(publisher.get_data()[::self.data_step, :])

    def connect(self, subscriber):
        return self._signals.connect(subscriber)

    def disconnect(self, subscriber=None):
        return self._signals.disconnect(subscriber)


class DataSkip(DataSubscriber, DataPublisher):
    def __init__(self, step):
        """Only sends every 'step'  data point"""
        DataSubscriber.__init__(self)
        DataPublisher.__init__(self)

        self._step = step

    def cb_new_data(self, publisher):
        data = publisher.get_data()
        self.raise_event(data[::self._step, :])


class Buffer(traitlets.HasTraits, traitlets.TraitType):
    input_data = ListofDict(default_value=ListofDict())
    output_data = ListofDict(default_value=ListofDict())
    buffer_full = traitlets.CBool(default_value=False)
    n_buffer = traitlets.CInt(default_value=0)

    def __init__(self, n: int, signal_keys: SignalKeyList = None):
        """ Buffer object

        :param shape: buffer shape tuple indicating the size of the buffer (n_data, n_dim)
        :type shape: tuple
        :param dtype:
        :type dtype: type
        """
        if signal_keys is None:
            signal_keys = SignalKeyList()

        super().__init__()

        self._buffer = ListofDict([{key: 0.0 for key in signal_keys} for _ in range(n)])
        self._keys = signal_keys
        self._n = n
        self._next_entry = 0
        self.buffer_full = False

        logger.info(f'Created buffer {id(self)}')

    @traitlets.observe('input_data')
    def _input_change(self, change):
        logger.info(f"{type(self).__name__ } ({id(self)}) received input change from"
                    f" {type(change.owner).__name__} ({id(change.owner)})")
        self.add_data(change.new)

    def add_data(self, data: ListofDict):

        # Verify keys:
        if not all(key in data.get_keys() for key in self._keys):
            return
        else:
            n_samples = len(data)
            next_entry = self._next_entry

        # To avoid performance loss, data is not moved around inside the buffer
        # instead we keep track of the latest entry of the buffer (next_entry)

        # Check where to add data in buffer array:
        if n_samples >= self._n:
            # Number of new samples is larger than buffer
            # Replace full buffer using the latest samples
            self._buffer = data[-self._n:]
            self._next_entry = 0
        else:
            # Less samples than buffer size, check where to add:
            if (next_entry + n_samples) > self._n:
                # samples are crossing border:
                n_bremainder = self._n - next_entry
                n_dremainder = n_samples - n_bremainder

                # Assign data:
                if n_bremainder > 0:
                    self._buffer[next_entry:] = data[:n_bremainder]
                if n_dremainder > 0:
                    self._buffer[:n_dremainder] = data[n_bremainder:]

                self._next_entry = n_dremainder
            else:
                self._buffer[next_entry:(next_entry+n_samples)] = data
                self._next_entry = next_entry + n_samples

        # Check buffer size:
        if not self.buffer_full:
            # update buffer size:
            self.n_buffer = min(self.n_buffer+n_samples, self._n)
            if self.n_buffer == self._n:
                self.buffer_full = True
        tmp = self.get_n_latest()[:]
        logger.info(f'Sending output from {type(self).__name__} ({id(self)})')
        self.output_data = tmp

    @property
    def data(self):
        return self.get_n_latest()

    @property
    def buffer_size(self):
        return self._n

    def get_n_latest(self, n=None):
        """ Returns n_latest samples. If n is None, it returns all samples in buffer
        """

        # Limit requested size to buffer size:
        n = self.n_buffer if n is None else min(n, self.n_buffer)

        if (self._next_entry-n) <0:
            diff = self._next_entry-n
            data = self._buffer[diff:] + self._buffer[:self._next_entry]
            if len(data) != n:
                raise ValueError("Dimensions not cool: {0}".format(data.shape))
        else:
            data = self._buffer[(self._next_entry-n):self._next_entry]

        return data

    def reset(self):
        self._next_entry = 0
        self.buffer_full = False


class SignalPublisher():
    pass


class DataBuffer(DataSubscriber, DataPublisher):

    def __init__(self, buffer_size, n_signals):
        DataSubscriber.__init__(self)
        DataPublisher.__init__(self)

        self._buffer = Buffer((buffer_size, n_signals))

    @property
    def buffer_full(self):
        return self._buffer.buffer_full

    @property
    def n_signals(self):
        """Number of signals in the buffer"""
        return self._buffer.shape[1]

    @property
    def buffer_size(self):
        """Total size of the buffer"""
        return self._buffer.shape[0]

    @property
    def n_samples(self):
        """Number of samples in buffer"""
        return self._buffer._n_buffer

    @property
    def data(self):
        return self._buffer.data

    def reset(self):
        self._buffer.reset()

    def add_data(self, data):
        self._buffer.add_data(data)

        # Notify subscribers of new data in buffer
        self.raise_event(self.data)

    def cb_new_data(self, publisher):
        data = publisher.get_data()
        self.add_data(data)

class SignalCreator(DataSubscriber, SignalPublisher):

    def __init__(self, buffer_size=1000, data_step=2):
        """Signal creator does the following:
        * Buffer data from a raw data stream such as Madern ni_task;
        * Creates signals from the buffered data (e.g. (data[:,0], data[:,1]), (data[:,0], data[:,2]),
        * Down-sample data stream (through the data_step)

        """
        DataSubscriber.__init__(self)
        SignalPublisher.__init__(self)

        # Allocate variables:
        self._buffer = DataBuffer(buffer_size=buffer_size, n_signals=1)
        self._signal_selector = SignalSelector(data_step=data_step)
        self._internal_subscriber = SimpleEventSubscriber(self._cb_signal_collector)
        self._initialized = False

    @property
    def data_step(self):
        """Data point interval to select"""
        return self._signal_selector.data_step

    @property
    def signal_selector(self):
        return self._signal_selector

    @property
    def signals(self):
        return self._signal_selector.signals

    def _connect_events(self):
        self._buffer.connect(self._signal_selector)
        self._signal_selector.connect(self._internal_subscriber)

    def _cb_signal_collector(self, publisher):
        self.raise_event(publisher.get_data())

    def cb_new_data(self, publisher):
        if not self._initialized:
            _, n_signals = publisher.get_data().shape
            self._buffer = DataBuffer(self._buffer.buffer_size, n_signals=n_signals)
            self._connect_events()
            self._initialized = True
            self.cb_new_data(publisher)
        else:
            self._buffer.cb_new_data(publisher)

    def reset(self):
        self._buffer.reset()

class DataGenerator(DataPublisher):

    def __init__(self, n_signals, pub_rate, n_samples):
        """

        :param n_signals: Number of signals to generate
        :param pub_rate: Publication rate
        :param n_samples:  Number of samples per publication
        """
        DataPublisher.__init__(self)

        self._running = False
        self._thrd = thrd.Thread()
        self.rate = pub_rate
        self._t0 = 0.0
        self._n_signals = n_signals
        self.n_samples = n_samples
        self.data_type = 'sine'

    @property
    def n_signals(self):
        return self._n_signals + 1

    def start(self):
        self._thrd = thrd.Thread(target=self.worker)
        self._running = True
        self._thrd.start()

    def stop(self):
        self._running = False

    def _func(self, t):
        if self.data_type == 'random':
            data = np.random.randn(self.n_samples, self._n_signals)
        else:
            data = (np.ones((self._n_signals, self.n_samples))*(np.sin(t*2*np.pi*300)+np.sin(t*2*np.pi*30))).T

        return data

    def worker(self):
        t_start = time.time()
        while self._running:
            t_now = time.time()
            t = self._t0 + np.arange(self.n_samples)/(self.rate*self.n_samples)
            samp = np.hstack([t[:, None], self._func(t)])
            self._t0 = t[-1]

            # Notify:
            now = time.time()
            self.raise_event(samp)
            dur = time.time() - now

            # sleep remaining time:
            t_diff = max(1.0/self.rate - (time.time() - t_now), 0.0)
            time.sleep(t_diff)

if __name__ == "__main__":

    # Testing Signal object and list
    mylist = SignalList()
    mylist.append(Signal('test1', 'unit', 0))
    mylist.append(Signal('test2', 'unit', 0))
    mylist.append(Signal('test3', 'unit', 0))
    for s in mylist:
        print(s.task_name)

    for s in mylist:
        mylist.remove(s)

    # Create data Generator:
    print('- Creating signal generator')
    mygen = DataGenerator(n_signals=4, pub_rate=20, n_samples=100)
    mytime_out = Timeout()                               #
    mygen.start()

    # Test signal:
    print('---- Testing Signal ----')
    signal_reader = SimpleEventSubscriber(lambda pub: print(pub.get_data()[0, :]))
    s = Signal('value 1', 'MyUnit', data_index=[0, 1])   # Create signal
    mygen.connect(s)                      # Connect to data generator
    s.connect(signal_reader)                             # Connect signal to reader
    mytime_out.wait_for_timeout(2)                       # wait for timeout
    mygen.disconnect()                    # Disconnect to stop printing
    print('* Testing Signal')

    # Signal Selector
    print('---- Signal Selector ----')
    signal_sel = SignalSelector(1)    # Signal creator
    signal_sel.add_selection([0, 1])  # Add signal combination
    signal_sel.add_selection([0, 2])  # Add signal combination

    # Create subscriber to display signals
    signal_reader = SimpleEventSubscriber(lambda pub: print([item.value for item in pub.get_data()]))

    # make connections
    mygen.connect(signal_sel)  # Connect generator to signal selector
    signal_sel.connect(signal_reader)         # Connect signal generator to the signal reader
    mytime_out.wait_for_timeout(2)            # Set timeout to do publishing
    mygen.disconnect()                        # Disconnect stop publishing

    # Buffer
    print('---- Data Buffer ----')
    my_buffer = DataBuffer(buffer_size=4, n_signals=mygen.n_signals)  # Buffer
    buffer_reader = SimpleEventSubscriber(lambda pub: print('data shape: {0}'.format(pub.get_data().shape)))

    print('-----Test Timeout-----')
    mygen.connect(my_buffer)
    my_buffer.connect(buffer_reader)
    mytime_out.wait_for_timeout(10)          # Set timeout to do publishing
    time.sleep(1)
    mygen.disconnect()                       # Disconnect stop publishing
