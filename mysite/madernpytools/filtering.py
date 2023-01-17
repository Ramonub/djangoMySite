import os, logging, traitlets, asyncio, sys
from threading import Lock
import numpy as np

import madernpytools.tools.utilities as mutils
import madernpytools.backbone as mbb
from madernpytools.tools.frequency_response import LowPassFilter, ButterWorthFilter
from madernpytools.backbone import IProcessStatus
from madernpytools.signal_handling import ISignalProvider, ISignalKeyList, SignalKeyList

logger = logging.getLogger(name=f'madernpytools.{__name__}')

class ClassFactory(mbb.IClassFactory):
    """ Factory class which allows to generate class instances from this module

    """

    @staticmethod
    def get(name):
        if name in sys.modules[__name__].__dict__:
            return eval(name)
        else:
            for mod in [mutils]:
                if name in getattr(mod, '__dict__'):
                    return getattr(mod, 'ClassFactory').get(name)
            raise TypeError('Could not find {}'.format(name))


class IFilter(mbb.ProcessStatus, ISignalProvider, mbb.TraitsXMLSerializer, traitlets.TraitType, mbb.HasTraitLinks):
    input_data = mutils.ListofDict(info_text='Filter input')
    output_data =mutils.ListofDict(info_text='Filter output')
    status = mbb.ProcessStatus(info_text='Filter status information')
    verbose = traitlets.Bool(default_value=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @traitlets.default('input_data')
    def _default_input(self):
        return mutils.ListofDict(items=[])

    @traitlets.default('status')
    def _default_status(self):
        return mbb.ProcessStatus()

    @traitlets.default('output_data')
    def _default_output(self):
        return mutils.ListofDict(items=[])

    def cb_inputdata(self, change):
        raise NotImplementedError()

    def cb_outputdata(self, change):
        raise NotImplementedError()


class DummyFilterItem(IFilter):

    def __init__(self, input_keys: ISignalKeyList=None, added_keys: ISignalKeyList=None):
        super().__init__()

        if input_keys is None:
            input_keys=SignalKeyList()
        if added_keys is None:
            added_keys=SignalKeyList()

        self._input_keys = input_keys
        self._added_keys = added_keys

    @property
    def added_keys(self) -> ISignalKeyList:
        return self._added_keys

    @property
    def required_input_keys(self) -> ISignalKeyList:
        return self._input_keys


class IFilterList(IFilter):
    sampling_rate = traitlets.CInt(default_value=1)
    input_keys = ISignalKeyList(default_value=[])

    def append(self, item):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __setitem__(self, index, item):
        raise NotImplementedError()

    def remove(self, item):
        raise NotImplementedError()

    def insert(self, index, item):
        raise NotImplementedError()

    @property
    def required_input_keys(self) -> ISignalKeyList:
        raise NotImplementedError()

    @property
    def added_keys(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class InputOutputValidator(object):

    @staticmethod
    def get_output_signals(filter_list: IFilterList) -> ISignalKeyList:
        if len(filter_list) > 0:
            # We assume the input_keys are available:
            key_list = filter_list[0].required_input_keys

            # Check per filter if the keys are valid:
            for f in filter_list:
                key_list += f.added_keys
        else:
            key_list = SignalKeyList()

        return key_list

    @staticmethod
    def validate_filter_list(filter_list: IFilterList) -> str:
        if len(filter_list) > 0:
            # We assume the input_keys are available:
            key_list = filter_list[0].required_input_keys

            # Check per filter if the keys are valid:
            for f in filter_list:
                # Verify keys:
                for key in f.required_input_keys:
                    if not key in key_list:
                        return '{} not in {}'.format(key, key_list)

                # Keys of this filter are valid, add 'added_keys' to key_list for next filter validation
                key_list += f.added_keys

        return None


class InputOutputValidator(object):

    @staticmethod
    def get_output_signals(filter_list: IFilterList) -> ISignalKeyList:
        if len(filter_list) > 0:
            # We assume the input_keys are available:
            key_list = filter_list[0].required_input_keys

            # Check per filter if the keys are valid:
            for f in filter_list:
                key_list += f.added_keys
        else:
            key_list = SignalKeyList()

        return key_list

    @staticmethod
    def validate_filter_list(filter_list: IFilterList) -> str:
        if len(filter_list) > 0:
            # We assume the input_keys are available:
            key_list = filter_list[0].required_input_keys

            # Check per filter if the keys are valid:
            for f in filter_list:
                # Verify keys:
                for key in f.required_input_keys:
                    if not key in key_list:
                        return '{} not in {}'.format(key, key_list)

                # Keys of this filter are valid, add 'added_keys' to key_list for next filter validation
                key_list += f.added_keys

        return None


class FilterList(IFilterList):

    _filter_list = traitlets.List()

    def __init__(self, items=None, sampling_rate=1, **kwargs):
        """ Allows to apply a series of filters to line profiles. This list automatically links the input and output
        of filters in list (according to list order). The first and last filter in list are connected to the input and
        output of the filter list.

        The FilterList also combines (active) state, progress and status messages of the filters it holds. The combined values can be used
        through the IProcessStatus interface

        :param items:
        :param kwargs:
        """
        self._lock = Lock()
        if items is None:
            items = []

        self._filter_links = []

        # Setup process linker for different filters:
        self._process_merger = mbb.ProcessStatusMerger()
        self._process_merger.link_progress_with(self)

        # Note that initialization order is imporant: _filter_list should be initinalized after input/output/sampling_rate
        # Not doing so, will cause conflicts with linking of the _filter_list
        super().__init__(input_data=mutils.Listof(),
                         output_data=mutils.Listof(),
                         sampling_rate=sampling_rate,
                         _filter_list=items,
                         var_names_mapping=[('items', '_filter_list'),
                                            ('sampling_rate', 'sampling_rate')],
                         **kwargs)

        if len(items) > 0:
            self._link_filters(self._filter_list)

    @traitlets.observe('_filter_list')
    def _filter_list_change(self, change):
        # Remove observers from old:
        if isinstance(change.old, list):
            self._unlink_filters(change.old)
            for f in change.old:
                self._process_merger.unobserve_process(f)

        # Add observers to new
        for f in change.new:
            self._process_merger.observe_process(f)

        # Link filters
        if len(change.new) > 0:
            self._link_filters(self._filter_list)

    def _validate_filterlist(self):
        flist = [DummyFilterItem(input_keys=self.required_input_keys)] + self._filter_list
        ret = InputOutputValidator.validate_filter_list(flist)
        if ret:
            raise RuntimeError('Filter list input/outputs do not match: '.format(ret))

    async def apply_filter(self):
        """

        :return:
        """
        # Input data is directly linked to first filter step: see traitlet links in the constructor
        pass

    def cb_inputdata(self, change):
        self.input_data = change.new

    def cb_outputdata(self, change):
        self.output_data = change.new

    def _unlink_filters(self, filter_list: list):
        # Unlink existing:
        if len(self._filter_links) > 0:
            for l in self._filter_links:
                l.unlink()
        self._filter_links = []

        # Remove observations:
        if isinstance(filter_list, list) and len(filter_list) > 0:
            self.unobserve(filter_list[0].cb_inputdata, 'input_data')
            filter_list[-1].unobserve(self.cb_outputdata, 'output_data')

            for i, filter in enumerate(filter_list[1:]):
                # Set observer (one-way street input to output:
                filter_list[i].unobserve(filter.cb_inputdata, 'output_data')

    def _link_filters(self, filter_list: list):
        """ Method links filters input and output in list order.
        First filter is connected to list input, Last filter is connected to list output

        :return:
        """

        # One-way street
        if isinstance(filter_list, list) and len(filter_list) > 0:
            filter_list[-1].observe(self.cb_outputdata, 'output_data')
            self.observe(filter_list[0].cb_inputdata, 'input_data')

            # Bind input-output of subsequent filters:
            for i, filter in enumerate(self._filter_list[1:]):
                # Set observer (one-way street input to output:
                self._filter_list[i].observe(filter.cb_inputdata, 'output_data')

                # Link sampling rate:
                if isinstance(filter, AbstractHasLowPassFilter):
                    l = traitlets.link(source=(self, 'sampling_rate'),
                                       target=(filter, 'sampling_rate'))
                    self._filter_links.append(l)

    def append(self, item):
        self._unlink_filters(self._filter_list)

        self._filter_list.append(item)
        self._validate_filterlist()

        self._link_filters(self._filter_list)
        self._process_merger.observe_process(item)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FilterList(items=self._filter_list[item], sampling_rate=self.sampling_rate)
        elif isinstance(item, int):
            return self._filter_list[item]

    def __setitem__(self, index, item):
        self._unlink_filters(self._filter_list)

        self._filter_list[index] = item
        self._validate_filterlist()

        self._link_filters(self._filter_list)
        self._process_merger.observe_process(item)

    def remove(self, item):
        self._unlink_filters(self._filter_list)

        self._filter_list.remove(item)
        self._validate_filterlist()

        self._link_filters(self._filter_list)
        self._process_merger.unobserve_process(item)

    def insert(self, index, item):
        self._unlink_filters(self._filter_list)

        self._filter_list.insert(index, item)
        self._validate_filterlist()

        self._link_filters(self._filter_list)
        self._process_merger.observe_process(item)

    @property
    def required_input_keys(self) -> ISignalKeyList:
        if len(self._filter_list) > 0:
            required_keys = SignalKeyList()     # Keep track of keys which are not supplied by filter list
            available_keys = SignalKeyList()    # Keep list of all available keys (i.e. keys added by individual filter steps)
            for f in self._filter_list:
                for key in f.required_input_keys:
                    if not key in available_keys:
                        required_keys.append(key)
                    available_keys += f.added_keys
            return required_keys
        else:
            return SignalKeyList()

    @property
    def added_keys(self):
        added_keys = SignalKeyList()
        for f in self._filter_list:
            added_keys += f.added_keys
        return added_keys

    def __len__(self):
        return len(self._filter_list)

    def __iter__(self):
        return iter(self._filter_list)


class AbstractFilter(IFilter):
    _update_request = traitlets.Bool(default_value=False)

    def __init__(self, parallel_processing=True, *args, **kwargs):
        self._lock = asyncio.Lock()
        self._parallel = parallel_processing
        super().__init__(*args, **kwargs)

        # Create links:
        self._status_links = []
        for attr_name in ['progress', 'error', 'status_message', 'active']:
            self._status_links.append(traitlets.link((self.status, attr_name), (self, attr_name)))

    @traitlets.observe('status')
    def _status_change(self, change):
        self.update_links(src_obj=change['new'])

    @traitlets.observe('active')
    def _active_change(self, change):
        pass

    @traitlets.observe('input_data')
    def _data_change(self, change):
        if len(self.input_data) > 0:
            # Set update request
            self._update_request = True
            if self._parallel:
                if not self.active:
                    self.active = True
                    asyncio.create_task(self._filter())
                else:
                    pass
            else:
                self._filter()

    async def _filter(self):
        # Set interaction:
        #with self._lock:
        self.active = True

        # Start filter:
        while self._update_request:
            self._update_request = False
            await self.apply_filter()

        #with self._lock:
        self.active = False

    async def apply_filter(self):
        raise NotImplementedError()

    @traitlets.observe('output_data')
    def _output_change(self, change):
        l_in = len(self.input_data)
        l_out = len(self.output_data)
        if l_in > 0:
            logger.info('{} filtered {} -> {}'.format(type(self).__name__,
                                                      l_in, l_out))

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

    def cb_inputdata(self, change):
        self.input_data = change.new

    def cb_outputdata(self, change):
        self.output_data= change.new


class AbstractHasLowPassFilter(traitlets.HasTraits):
    sampling_rate = traitlets.CInt(default_value=1)

    def __init__(self, *args, **kwargs):
        # Filter list and link list:
        self._filter_list = []
        super().__init__(*args, **kwargs)

    def _append_bw_filter(self, item: LowPassFilter):
        """ Add Butterworth filter to filter list

        :param item:
        :return:
        """
        # Create link
        # Make sure that filter frequency is acceptable for the new sampling rate:
        # TODO: This is a bit of a dirty hack, as the implementations of ButterWorthFilter (Lowpass, Bandpass, and HighPass)
        # do not respond to changes in the frequency value
        if self.sampling_rate/2 < item.low_pass_frequency:
            raise ValueError('Filter low-pass frequency ({}) should be smaller than half sampling rate ({})'.format(item.low_pass_frequency,self.sampling_rate))
            item.low_pass_frequency = self.sampling_rate/2 - 1e-3

        self._filter_list.append(item)

    def _remove_bw_filter(self, item: LowPassFilter):
        """ Remove Butterworth filter to filter list

        :param item:
        :return:
        """

        # Remove from item
        self._filter_list.remove(item)

    @traitlets.observe('sampling_rate')
    def _sampling_rate_change(self, change):
        for filter in self._filter_list:
            if change.new < filter.low_pass_frequency / 2:
                filter.low_pass_frequency = change.new / 2 - 1e-3
            filter.fs = change.new


class NumpyOperation(mbb.TraitsXMLSerializer, ISignalProvider):
    name = traitlets.CUnicode()
    function = traitlets.TraitType()
    argument_keys = traitlets.List(default_value=[])

    def __init__(self, name: str, function: np.ufunc, argument_keys: list):
        super().__init__(name=name, function=function, argument_keys=argument_keys)

    def apply(self, input_data: mutils.ListofDict):
        return self.function(*[input_data.get_key_values(key) for key in self.argument_keys])

    @property
    def required_input_keys(self) -> ISignalKeyList:
        return SignalKeyList(items=self.argument_keys)

    @property
    def added_keys(self) -> ISignalKeyList:
        raise NotImplementedError()


class ZeroSignals(AbstractFilter):
    signal_keys = SignalKeyList()

    def __init__(self, signal_keys:SignalKeyList=None):

        if signal_keys is None:
            signal_keys = SignalKeyList()

        self._zero_values = {}
        super().__init__(signal_keys=signal_keys)

    @traitlets.observe('signal_keys')
    def _signals_change(self, change):
        """ Handle change list

        @param change:
        @return:
        """

        zero_values = {}
        if isinstance(change.old, mutils.ListofDict):
            for key in change.new:
                if key in change.old:
                    zero_values[key] = change.old[key]
                else:
                    zero_values[key] = 0.0
        else:
            zero_values = {key: 0.0 for key in change.new}
        self._zero_values = zero_values

    def zero_signals(self):
        """

        @return:
        """
        # Update mean values:
        data = self.input_data[:]
        for key in self.signal_keys:
            self._zero_values[key] = np.mean(data.get_key_values(key))

    async def apply_filter(self):
        """

        @return:
        """
        data = self.input_data[:]

        for item in data:
            for key, value in self._zero_values.items():
                item[key] -= value

        self.output_data = data

    @property
    def required_input_keys(self) -> ISignalKeyList:
        return self.signal_keys

    @property
    def added_keys(self) -> ISignalKeyList:
        return SignalKeyList([])


class MathOperation(AbstractFilter):
    filter_operations = traitlets.List()

    def __init__(self, filter_operations: list):
        super().__init__(filter_operations=filter_operations)

    @traitlets.validate('filter_operations')
    def _filter_operations(self, proposal):
        for item in proposal['value']:
            if not isinstance(item, NumpyOperation):
                raise traitlets.TraitError('Expected list of {}'.format(NumpyOperation.__name__))
        return proposal['value']

    async def apply_filter(self):
        data = self.input_data[:]

        # Compute results:
        results = {}
        for operation in self.filter_operations:
            args = [data.get_key_values(key) for key in operation.argument_keys]
            results[operation.name] = operation.function(*args)

        try:
            # Assign results
            for i, item in enumerate(data):
                data[i] = item | {key: values[i] for key, values in results.items()}
        except Exception as e:
            logger.error(e)

        self.output_data = data

    @property
    def required_input_keys(self) -> ISignalKeyList:
        # Collect arguments from keys:
        key_list = SignalKeyList([])
        for item in self.filter_operations:
            key_list += item.required_input_keys

        return key_list

    @property
    def added_keys(self) -> ISignalKeyList:
        return SignalKeyList([f.name for f in self.filter_operations])


class DataSelector(AbstractFilter):
    selected_keys = SignalKeyList(default_value=SignalKeyList([]))
    available_keys = SignalKeyList(default_value=SignalKeyList([]))
    enabled = traitlets.Bool(default_value=True)

    def __init__(self, available_keys: SignalKeyList=None, selected_keys: SignalKeyList=None, enabled: bool=True):
        """ Filter removes non-selected signals

        """

        if selected_keys is None:
            selected_keys = SignalKeyList([])

        if available_keys is None:
            # Copy atleast selected keys
            available_keys = selected_keys

        super().__init__(input_data=mutils.ListofDict(items=[]),
                         output_data=mutils.ListofDict(items=[]),
                         available_keys=available_keys,
                         selected_keys=selected_keys,
                         enabled=enabled,
                         var_names_mapping=[('selected_keys', 'selected_keys'),
                                            ('enabled', 'enabled')]
                         )

    @traitlets.observe('selected_keys', 'enabled')
    def _keys_changed(self, change):
        """

        :param change:
        :return:
        """
        self._data_change(change)

    @traitlets.validate('selected_keys')
    def _selected_keys(self, proposal):
        if all(key in self.available_keys for key in proposal['value']):
            return proposal['value']
        else:
            raise traitlets.TraitError('Selected keys are note a subset of available keys')

    @traitlets.observe('available_keys')
    def _available_keys(self, change):
        # Check current selected keys:
        new_selection = []
        for key in self.selected_keys:
            if key in change.new:
                new_selection.append(key)
        self.selected_keys = new_selection

    async def apply_filter(self):
        """

        :return:
        """
        input = self.input_data[:]

        if self.enabled:
            # Filtering:
            output = mutils.ListofDict()
            self.status_message = 'Selecting desired data'
            for i, p in enumerate(input):
                output.append({key: p[key] for key in self.selected_keys if key in p})
                if i % 5000:
                    self.progress = i / len(input)
            self.status_message = ''
            with self._lock:
                self.output_data = output
        else:
            # No filtering:
            with self._lock:
                self.output_data = input

    @property
    def required_input_keys(self):
        return self.selected_keys

    @property
    def added_keys(self):
        return ['!*'] + self.selected_keys


class AbstractDataProcessor(AbstractFilter):
    """ Data processor for a time-series of profile data used to detect the tooling convexity

    """

    filter_list = FilterList(default_value=FilterList())
    sampling_rate = traitlets.CInt(default_value=50)

    def __init__(self,
                 filter_list=None,
                 sampling_rate=50,
                 verbose=False,
                 **kwargs,
                 ):
        """

        :param filter_list:  Filterlist
        :param verbose: Filter verbosity
        :param f_height: Height function
        """

        if filter_list is None:
            filter_list = FilterList

        names = ['filter_list', 'sampling_rate']  # Names to link on xml serializing
        if 'var_names_mapping' in kwargs:
            var_names_mapping = kwargs['var_names_mapping']
            var_names_mapping += list(zip(names, names))
        else:
            var_names_mapping = list(zip(names, names))

        self._process_merger = mbb.ProcessStatusMerger()
        self._process_merger.link_progress_with(self)

        # Linking keys:
        self._filter_link_keys = ['sampling_rate', 'input_data', 'output_data']
        self._filter_links = {}

        # Initialize sub-classes:
        names = ['filter_list', 'sampling_rate']   # Names to link on xml serializing

        super().__init__(
            # Filter outputs:
            filter_list=filter_list,
            verbose=verbose,
            sampling_rate=sampling_rate,
            var_names_mapping=var_names_mapping
        )

    @property
    def output_keys(self):
        # Get unique output values:
        f_list = [DummyFilterItem(input_keys=self.required_input_keys)] + [f for f in self.filter_list]
        return InputOutputValidator.get_output_signals(f_list)

    @traitlets.validate('filter_list')
    def _filter_list_validate(self, proposal):
        """ Validate the content of the proposed filter list

        :param proposal:
        :return:
        """

        # Check if all filter items inherit from AbstractFilter
        for item in proposal['value']:
            if not isinstance(item, AbstractFilter):
                raise traitlets.TraitError('Supplied filter list contains a non-AbstractFilter item: {}'.format(type(item)))

        # No failure, return proposal
        return proposal['value']

    @traitlets.observe('filter_list')
    def _filter_list_change(self, change):
        # Observe filter list:
        self._process_merger.unobserve_process(change.old)
        self._process_merger.observe_process(change.new)

        for key, l in self._filter_links.items():
            l.unlink()

        self._filter_links = {key: traitlets.link((self, key), (change['new'], key)) for key in self._filter_link_keys}

    async def _filter(self):
        self.active = True
        self.status_message = 'Data processor started...'

        # Start filter:
        while self._update_request:
            self._update_request = False
            await self.apply_filter()

        if not self.filter_list.active and self.active:
            # Reset active
            self.active=False

    async def apply_filter(self):
        """

        :return:
        """
        # This filter actually does not
        raise NotImplementedError()

    @property
    def required_input_keys(self):
        raise NotImplementedError()

    @property
    def added_keys(self):
        return self.filter_list.added_keys
