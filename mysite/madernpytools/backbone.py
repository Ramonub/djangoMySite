import time, abc
import threading as thrd
import traitlets

from threading import Lock

import numpy as np
from enum import Enum
import xml.etree.cElementTree as ET


class AbstractEventSubscriber(object):
    pass


class HasTraitLinks(object):

    def __init__(self):
        """ Provides class with method to update existing links

        """
        self._trait_links = []

    def update_links(self, src_obj=None, trg_obj=None, list_attribute_name='_trait_links'):
        """ Updates source or target objects of links specified in the specified list_attribute_name

        @param src_obj: New link_progress source object, if not specified existing source is used
        @param trg_obj: New link_progress target object, if not specified existing target is used
        @param list_attribute_name: attribute task_name of the list for which links need to be updated, defaults to '_trait_links'
        @return:
        """
        if hasattr(self, list_attribute_name):
            new_links = []
            for old_link in getattr(self, list_attribute_name):
                # Unlink to prevent synchronization issues between old and new values:
                old_link.unlink()

                # Create new link:
                if isinstance(src_obj, traitlets.HasTraits) and isinstance(trg_obj, traitlets.HasTraits):
                    new_link = traitlets.link((src_obj, old_link.source[1]), (trg_obj, old_link.target[1]))
                elif isinstance(src_obj, traitlets.HasTraits):
                    new_link = traitlets.link((src_obj, old_link.source[1]), old_link.target)
                elif isinstance(trg_obj, traitlets.HasTraits):
                    new_link = traitlets.link(old_link.source, (trg_obj, old_link.target[1]))
                else:
                    # Keep old link_progress
                    new_link = traitlets.link(old_link.source, old_link.target)
                # add new link_progress & Unlink old:
                new_links.append(new_link)

            setattr(self, list_attribute_name, new_links)


class IProcessStatus(traitlets.HasTraits, traitlets.TraitType):
    """ Standard process progress indicators

    """
    progress = traitlets.CFloat(default_value=0.0, help='Process progress (0 - 1 )')
    error = traitlets.CInt(default_value=-1, help='Error value')
    status_message = traitlets.CUnicode(default_value='', help='Message describing process status')
    active = traitlets.CBool(default_value=False, help='Indication if process is active')


class ProcessStatus(IProcessStatus):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._process_status_links = {}

    @traitlets.validate('progress')
    def _validate_progress(self, proposal):
        if (0 <= proposal['value']) and (proposal['value'] <= 1.0):
            return proposal['value']
        else:
            raise traitlets.TraitError('Progress value should lie in range [0,1], received {}'.format(proposal['value']))

    @traitlets.default('progress')
    def _default_progress(self):
        return 0.0

    @traitlets.default('error')
    def _default_error(self):
        return -1

    @traitlets.default('status_message')
    def _default_statusmessage(self):
        return ''

    @traitlets.default('active')
    def _default_active(self):
        return False

    def link_progress_with(self, other: IProcessStatus):

        if other in self._process_status_links:
            raise RuntimeWarning('Other already linked')
            return self._process_status_links[other]
        else:
            # Create links
            new_links = []
            for key in ['progress', 'error', 'status_message', 'active']:
                new_links.append(traitlets.link((self, key), (other, key)))
            self._process_status_links[other] = new_links
            return self._process_status_links[other]

    def unlink_progress_with(self, other: IProcessStatus = None):
        if other is None:
            for item in self._process_status_links:
                self.unlink_progress_with(item)
        elif other in self._process_status_links:
            links = self._process_status_links.pop(other)
            for l in links:
                l.unlink()
        else:
            raise RuntimeWarning('Could not find links to object')

    def progress_links(self):
        return self._process_status_links


class IProcessStatusMerger(ProcessStatus):

    def observe_process(self, item):
        raise NotImplementedError()

    def unobserve_process(self, item):
        raise NotImplementedError()


class MadernObject(traitlets.HasTraits, traitlets.TraitType):
    illustration_path = ''

    def __eq__(self, other):
        """

        @param other:
        @return:
        """
        if type(self) is type(other):
            return all([getattr(other, key) == val for key, val in self.__dict__['_trait_values'].items()])
        else:
            return False

    def __str__(self):
        return f'Madern Object {type(self)}:\n' + ''.join([f'{key:<15}: {value}\n'
                                                        for key, value in self.__dict__['_trait_values'].items()])


class ProcessStatusMerger(IProcessStatusMerger):
    verbose = traitlets.Bool(default_value=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observed_items = []
        self._verbosity_links = {}
        self._lock = Lock()

    def observe_process(self, item):
        """Implement the new filterlist in the data processor

        :param change:
        :return:
        """
        self._observed_items.append(item)

        # Reset filter progress observers::
        item.observe(self._progress_change, 'progress')
        item.observe(self._message_change, 'status_message')
        item.observe(self._active_change, 'active')

        # Link verbosity:
        self._verbosity_links[item] = traitlets.link((self, 'verbose'), (item, 'verbose'))

    def unobserve_process(self, item):
        if item in self._observed_items:
            self._observed_items.remove(item)
            item.unobserve(self._progress_change)
            item.unobserve(self._message_change)
            item.unobserve(self._active_change)

        # Remove verbosity link:
        if item in self._verbosity_links:
            self._verbosity_links[item].unlink()

    def _active_change(self, change):
        """ Respond to state change of the 'active' property of filters in the filter_list

        We merge the active properties of the filter_list items in 'or' fashion

        :param change:
        :return:
        """
        with self._lock:
            vals = [item.active for item in self._observed_items]
            self.active = np.sum(vals) > 0


    def _progress_change(self, change):
        """ Respond to change of the 'progress' property of filters in the filter_list

        We merge the filter progresses as a weighted combination
        :param change:
        :return:
        """
        with self._lock:
            val = 0
            N = len(self._observed_items)
            for item in self._observed_items:
                val += item.progress / N
            self.progress = min(val, 1.0)

    def _message_change(self, change):
        """ Respond to change of the 'status_message' property of filters in the filter_list

        Status messages of individual filters are stacked

        :param change:
        :return:
        """
        with self._lock:
            val = 0
            msg = ''
            for item in self._observed_items:
                if item.status_message != '':
                    msg += '{}, '.format(item.status_message)
            if msg != '':
                msg = 'Data processing: ' + msg[:-2].capitalize() + '...'

            self.status_message = msg

class MadernObject(traitlets.HasTraits, traitlets.TraitType):
    illustration_path=''

    def __eq__(self, other):
        if type(self) is type(other):
            return all([getattr(other, key) == val for key, val in self.__dict__['_trait_values'].items()])
        else:
            return False


class IClassFactory(object):
    """
    Class factory enables construction of class instances outside object scope
    """
    @staticmethod
    def get(name):
        return eval(name)


class IXML:
    """ XML interface

    """

    def to_xml(self):
        """Write object to xml element"""
        raise NotImplementedError()

    @staticmethod
    def from_xml(root, *args, **kwargs):
        """Load object from xml element"""
        raise NotImplementedError()


class AbstractXMLSerializer(object):

    @staticmethod
    def serialize_into_root(item, xml_root, tag):
        """  Serialize item in to xml_root.

        :param item: Object to serialize (Supports: dict(of), list(of), IXML, str, float, int)
        :param xml_root: root in which item should be put
        :param tag: tag of object in xml tree

        """

        # Create XML entry:
        xml_item = ET.SubElement(xml_root, tag)
        xml_item.set('Type', type(item).__name__)

        # Add contents depending on type:
        if isinstance(item, IXML):
            xml_item.append(item.to_xml())
        elif isinstance(item, (str, float, int, bool)):
            xml_item.text = str(item)
        elif isinstance(item, (list, tuple)):
            for i, sub_item in enumerate(item):
                AbstractXMLSerializer.serialize_into_root(sub_item, xml_item, 'index_{0}'.format(i))
        elif isinstance(item, dict):
            for key, sub_item in item.items():
                AbstractXMLSerializer.serialize_into_root(sub_item, xml_item, '{0}'.format(key))
        elif isinstance(item, np.ufunc):
            xml_item.text = repr(item).split('\'')[1]
        else:
            raise ValueError("Error in converting {0} of parameter {2}: Cannot convert {1} into XML format".format(type(item).__name__,
                                                                                                                   item, tag))

        return item

    @staticmethod
    def deserialize_xml_item(xml_item, class_factory: IClassFactory = None):
        """Deserialize xml_item using the specified class factory.

        The class factory ensures proper deserialization. It ensures the objects are iniatialized in the proper scope.

        : xml_item: XML item to deserialize
        : class_factory: class factory to use when performing serialization.

        """

        # Define class factory:
        class_factory = class_factory if class_factory is not None else IClassFactory

        try:
            type_str = xml_item.get('Type')
            if type_str in ['str', 'bool', 'list', 'dict', 'int', 'float', 'tuple']:
                # Default type:
                item_type = eval(type_str)
            elif type_str == 'ufunc':
                item_type = np.ufunc #('np.{}'.format(f_name))
            else:
                # 'Custom' Type:
                item_type = class_factory.get(type_str)

        except NameError as e:
            raise NameError("Could not find {0} for building {1}: {2}".format(xml_item.get('Type'), xml_item, e))

        # Extract dict:
        if issubclass(item_type, IXML):
            item = item_type.from_xml(xml_item.find(xml_item.get('Type')), class_factory)
        elif issubclass(item_type, dict):
            item = {}
            for child in xml_item:
                item[child.tag] = AbstractXMLSerializer.deserialize_xml_item(child, class_factory)
        elif issubclass(item_type, np.ufunc):
            item = eval('np.{}'.format(xml_item.text))
        elif issubclass(item_type, list):
            item = []
            for child in xml_item:
                item.append(AbstractXMLSerializer.deserialize_xml_item(child, class_factory))
        elif issubclass(item_type, tuple):
            item = []
            for child in xml_item:
                item.append(AbstractXMLSerializer.deserialize_xml_item(child, class_factory))
            item = tuple(item)
        elif issubclass(item_type, bool):
            item = xml_item.text.lower() == 'true'
        else:
            item = item_type(xml_item.text)

        return item


class TraitsXMLSerializer(traitlets.HasTraits, IXML, AbstractXMLSerializer):

    def __init__(self, class_factory: IClassFactory=None, var_names_mapping=None, *args, **kwargs):
        """
        XML serialization class for Traits. Inherit this class to enable XML serialization.

        :param var_names_mapping: Name of the root xml element (see below)

        Serialization relies on the var_names_mapping, a list of tuples which relates the internal variables from
        self.__dict__ to the arguments of the class.

        Example of usage:

        class Example(XMLSerializer):

            def __init__(self, a, b, c):

                self._a_variable = a
                self._b_variable = b
                self._c_variable = c

                # Define variable task_name mapping
                mapping = [('a', '_a_variable), ('b', '_b_variable), ... ]
                # or shorter, using some Python magic:
                mapping = list(zip(Example.__init__.__code__.co_varnames[1:], self.__dict__.keys()))

                XMLSerializer.__init__(self, mapping)

        # Generate object:
        my_example = Example(1,2,4)
        xml_data = my_example.to_xml()

        # Reconstruct from xml element:
        my_example2 = Example.from_xml(xml_data)

        """
        super().__init__(*args, **kwargs)

        # Define mapping for serialization:
        if var_names_mapping is None:
            self._var_names_mapping = list(zip(sorted(type(self).__init__.__code__.co_varnames[1:]),
                                               sorted(self.__dict__['_trait_values'].keys()))
                                           )
        else:
            self._var_names_mapping = var_names_mapping

        self._class_factory = class_factory

    @property
    def var_names_mapping(self):
        return self._var_names_mapping

    def to_xml(self):
        """Convert instance to XML"""

        xml_root = ET.Element(type(self).__name__)
        xml_root.set('Type', type(self).__name__)

        for constructor_key, trait_key in self._var_names_mapping:
            item = getattr(self, trait_key)
            TraitsXMLSerializer.serialize_into_root(item, xml_root, tag=constructor_key)

        return xml_root

    @staticmethod
    def from_xml(xml_element, class_factory: IClassFactory = None):
        """Initialize instance from XML data"""
        class_factory = class_factory if class_factory is not None else IClassFactory

        item_dict = {}
        for child in xml_element:
            item_dict[child.tag] = TraitsXMLSerializer.deserialize_xml_item(child, class_factory)

        # Construct object:
        item_type = class_factory.get(xml_element.get('Type'))

        try:
            return item_type(**item_dict)
        except TypeError as e:
            raise TypeError("Creation of {0} object failed: {1}".format(item_type, e))


class XMLSerializer(AbstractXMLSerializer, IXML):
    def __init__(self, var_names_mapping: list = None, class_factory: IClassFactory=None):
        """
        XML serialization class. Inherit this class to enable XML serialization.

        :param var_names_mapping: Name of the root xml element (see below)

        Serialization relies on the var_names_mapping, a list of tuples which relates the internal variables from
        self.__dict__ to the arguments of the class.

        Example of usage:

        class Example(XMLSerializer):

            def __init__(self, a, b, c):

                self._a_variable = a
                self._b_variable = b
                self._c_variable = c

                # Define variable task_name mapping
                mapping = [('a', '_a_variable), ('b', '_b_variable), ... ]
                # or shorter, using some Python magic:
                mapping = list(zip(Example.__init__.__code__.co_varnames[1:], self.__dict__.keys()))

                XMLSerializer.__init__(self, mapping)

        # Generate object:
        my_example = Example(1,2,4)
        xml_data = my_example.to_xml()

        # Reconstruct from xml element:
        my_example2 = Example.from_xml(xml_data)

        """

        # Define mapping for serialization:
        if var_names_mapping is None:
            n_args = type(self).__init__.__code__.co_argcount
            self._var_names_mapping = list(zip(type(self).__init__.__code__.co_varnames[1:n_args], self.__dict__.keys()))
        else:
            self._var_names_mapping = var_names_mapping

        self._class_factory = class_factory

    @property
    def var_names_mapping(self):
        return self._var_names_mapping

    def to_xml(self):
        """Convert instance to XML"""

        xml_root = ET.Element(type(self).__name__)
        xml_root.set('Type', type(self).__name__)

        for constructor_key, internal_key in self._var_names_mapping:
            item = self.__dict__[internal_key]
            TraitsXMLSerializer.serialize_into_root(item, xml_root, tag=constructor_key)

        return xml_root

    @staticmethod
    def from_xml(xml_element, class_factory: IClassFactory = None):
        """Initialize instance from XML data"""
        class_factory = class_factory if class_factory is not None else IClassFactory

        item_dict = {}
        for child in xml_element:
            item_dict[child.tag] = TraitsXMLSerializer.deserialize_xml_item(child, class_factory)

        # Construct object:
        item_type = class_factory.get(xml_element.get('Type'))

        try:
            return item_type(**item_dict)
        except TypeError as e:
            raise TypeError("Creation of {0} object failed: {1}".format(item_type, e))


class EventSubscriber(AbstractEventSubscriber):
    def update(self, publisher):
        raise NotImplementedError()


class AbstractEventPublisher(object):

    def __init__(self, data_type, subscriber_type=AbstractEventSubscriber, subscriber_delegate_attr='', **kwargs):

        # Event data
        self._data_type = data_type
        self._event_data = None

        # Subscribers restrictions
        self._subscribers = []
        self._subscriber_type = subscriber_type
        self._delegate_subscriber = subscriber_delegate_attr

    def connect(self, subscriber: AbstractEventSubscriber):
        """Connect refresh widget"""

        if isinstance(subscriber, AbstractEventSubscriber):
            _subscriber = getattr(subscriber, self._delegate_subscriber, subscriber)  # Potentially delegate ni_task
            self._subscribers.append(_subscriber)
        else:
            raise TypeError("Subscriber should be of type {0}".format(EventSubscriber.__name__))

    def disconnect(self, subscriber=None):
        """Disconnect refresh widget"""

        if isinstance(subscriber, AbstractEventSubscriber) or (subscriber is None):
            _subscriber = getattr(subscriber, self._delegate_subscriber, subscriber)  # Potentially delegate ni_task
            if _subscriber in self._subscribers:
                self._subscribers.remove(_subscriber)
            elif _subscriber is None:
                self._subscribers.clear()
        else:
            raise TypeError("Subscriber should be of type {0} or None".format(self._subscriber_type.__name__))

    def notify(self):
        for s in self._subscribers:
            s.update(self)

    def get_data(self):
        return self._event_data

    def raise_event(self, data=None):
        # Set event data:
        if (data is None) and (self._data_type is None):
            self.notify()
        elif not (self._data_type is None) and (data is None):
            raise RuntimeError('No data provided {0}'.format(data))
        elif isinstance(data, self._data_type):
            self._event_data = data
            self.notify()
        else:
            raise ValueError("Event data should be of type {0}".format(self._data_type.__name__))


class EventPublisher(AbstractEventPublisher):

    def __init__(self, data_type=None):
        AbstractEventPublisher.__init__(self, data_type=data_type,
                                        subscriber_type=EventSubscriber,
                                        subscriber_delegate_attr='')


class SimpleEventSubscriber(EventSubscriber):

    def __init__(self, h_callback, **kwargs):
        """ Simple event subscriber

        :param h_callback: Handle to callback function
        :param kwargs:     Keyword arguments to pass on event (optional)
        """
        self._callback = h_callback
        self._kwargs = kwargs

    def update(self, publisher):
        self._callback(publisher, **self._kwargs)


class DataSubscriber(AbstractEventSubscriber):
    def __init__(self, h_new_data=None):
        self.data_subscriber = SimpleEventSubscriber(self.cb_new_data)

        if not (h_new_data is None):
            # Set data subscriber:
            self.cb_new_data = h_new_data

    def cb_new_data(self, publisher):
        raise NotImplementedError()


class DataPublisher(AbstractEventPublisher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, data_type=np.ndarray,
                         subscriber_type=DataSubscriber,
                         subscriber_delegate_attr='data_subscriber',
                         **kwargs)


class JobState(Enum):
    Idle = 0
    Active = 1
    Done = 2
    Abort = 3


class JobResult(object):

    def __init__(self, job_state, data):
        self._state = job_state
        self._data = data

    @property
    def state(self):
        return self._state

    @property
    def data(self):
        return self._data


class BackgroundJob(EventPublisher):

    def __init__(self):
        EventPublisher.__init__(self, data_type=JobResult)
        self._state = JobState.Idle
        self._data = None

    @property
    def state(self):
        """ Job state
        :returns: State of the job
        :rtype: JobState
        """
        return self._state

    def abort(self):
        """ Abort job

        :return: None
        """
        self._state = JobState.Abort

    def do_work(self):
        """ Perform job

        :return:
        """
        self._state = JobState.Active
        self._work_sequence()
        if not (self._state == JobState.Abort):
            self._state = JobState.Done

        self.raise_event(self.get_result())

    def _work_sequence(self):
        raise NotImplementedError()

    def get_result(self):
        """
        :return:
        :rtype: JobResult
        """
        raise NotImplementedError()


class RepetitiveJob(BackgroundJob):

    def __init__(self, rate=100):
        """ Repetitive job.
            Take note: the rate is only an indication. The rate is controlled using time.sleep().
            This method is inaccurate under both windows and unix. Under windows rates are impossible as the sleep method
            does not facilitate short sleeps

        :param rate:
        """
        BackgroundJob.__init__(self)
        self._rate = rate

    # Properties:
    @property
    def rate(self):
        """
        :return: Background worker rate
        :rtype: int
        """
        return self._rate

    @rate.setter
    def rate(self, rate):
        """
        :param rate: Background worker rate
        :type rate: int
        :return:
        """
        self._rate = rate

    # Methods:
    def setup(self):
        """ Operations performed before repetitive job is started

        :return: None
        """
        pass

    def clean_up(self):
        """ Operations performed after repetitive job is finished

        :return: None
        """
        pass

    def work(self):
        """ Repetitive operation"""
        raise NotImplementedError()

    def _work_sequence(self):
        """ Setup job, perform job operation repetitively and cleanup after stop

        :return:
        """
        self.setup()

        while self.state == JobState.Active:
            t_start = time.time()

            # Do work:
            self.work()

            # Keep loop rate:
            t_end = time.time()
            dt = t_end - t_start
            t_sleep = max(0.0, 1.0/self._rate - dt)
            time.sleep(t_sleep)

        self.clean_up()

    def get_result(self):
        pass


class BackgroundWorker(EventPublisher):

    def __init__(self, job, rate=100):
        EventPublisher.__init__(self, JobResult)

        self._thrd = thrd.Thread()
        self._rate = rate
        self._job = BackgroundJob()

        self.job = job

    def start(self):
        if not (self._job.state == JobState.Active):
            self._thrd = thrd.Thread(target=self.do_work)
            self._thrd.start()

    @property
    def job(self):
        return self._job

    @job.setter
    def job(self, job):
        if issubclass(type(job), BackgroundJob):
            self._job = job
        else:
            raise ValueError("Job should be subclass of {0}".format(BackgroundJob.__name__))

    def stop(self):
        if self._job == JobState.Active:
            self._job.abort()

    def do_work(self):
        # Do work
        self._job.do_work()

        # Notify subscribers when finished:
        self.raise_event(self._job.get_result())


class JobQue(object):

    def __init__(self, n_threads):
        """Que of jobs

        :param n_threads:
        """
        self._n_threads = n_threads
        self._job_que = []
        self._workers = []
        self._lock = thrd.Lock()
        self._sub_worker_finished = SimpleEventSubscriber(h_callback=self.cb_job_finished)

        # Create threads:
        for n in range(n_threads):
            worker = BackgroundWorker(BackgroundJob())
            worker.connect(self._sub_worker_finished)
            self._workers.append(worker)

    def active_jobs(self):
        """

        :return: Number of active jobs
        """

        cnt = 0
        for w in self._workers:
            if w.job.state == JobState.Active:
                cnt += 1
        return cnt

    def __len__(self):
        return len(self._job_que)

    def __getitem__(self, index):
        return self._job_que[index]

    def pop(self, index):
        return self._job_que.pop(index)

    def cb_job_finished(self, publisher: BackgroundWorker):
        """

        :param publisher:
        :return:
        """

        # Assign new job
        self._lock.acquire()
        if len(self._job_que)>0:
            publisher.job = self._job_que.pop(0)
            publisher.start()
        self._lock.release()

    def is_running(self):
        """Returns true when jobs are being processed or in que

        :return:
        """
        if len(self)==0 and self.active_jobs()==0:
            return False
        else:
            return True

    def append(self, job: BackgroundJob):
        """

        :param job:  Job to add to que
        :return:
        """
        job_assigned = False

        # Check if we can assign it to an idle worker:
        self._lock.acquire()
        if self.active_jobs() < self._n_threads:
            # Try to assign job to idle worker:
            for w in self._workers:
                if not (w.job.state == JobState.Active):
                    w.job = job
                    w.start()

                    job_assigned = True
                    break

        # Add it to que if it is not assigned
        if not job_assigned:
            self._job_que.append(job)
        self._lock.release()


class AbstractArgumentVerifier(object):

    def verify(self, item):
        raise NotImplementedError()


class ArgumentVerifier(AbstractArgumentVerifier):

    def __init__(self, argument_class, default_value):
        """

        :param argument_class: type or tuple of allotted types
        :param default_value: value which Verifier returns on verification of None
        """
        self._argument_class = argument_class

        # Check if default value is of argument_class:
        self._default_value = None
        self._default_value = self.verify(default_value)

    def verify(self, item):
        """Verify if item meets Verifier requirements"""
        if item is None:
            # If supplied argument is None, return default value
            return self._default_value
        elif isinstance(item, self._argument_class):
            # If supplied argument is of required type, return supplied argument
            return item
        else:
            # If none of the above raise value error
            raise TypeError("Argument is of type {2} but requires type {1}".format(item, self._argument_class,
                                                                                   type(item)))


class AbstractDataProcessor(DataSubscriber):

    def __init__(self):
        DataSubscriber.__init__(self)

    @property
    def event_new_data(self):
        raise NotImplementedError

    def cb_new_data(self, publisher):
        raise NotImplementedError()


class Timeout(EventSubscriber):
    # MZ not finished, and not tested

    def __init__(self, dt=0.1):
        self._dt = dt
        self._thrd = thrd.Thread(target=self._wait_for_timeout)
        self._timeout_occurred = False
        self._running = False
        self._abort = False
        self._timeout = 10
        self._event_timeout = EventPublisher()
        self.condition = False

    @property
    def timeout(self):
        return self._timeout_occurred

    @property
    def event_timeout(self):
        return self._event_timeout

    def reset(self):
        if self._running:
            self._abort
        self._timeout_occurred = False
        self.condition = False
        self._running = False
        self._abort = False

    def start(self, timeout):
        """Start timeout"""
        self.reset()
        self._timeout = timeout
        self._thrd = thrd.Thread(target=self._wait_for_timeout)
        self._thrd.start()

    def abort(self):
        """Abort timeout"""
        self._abort = True
        self._thrd.join()

    def wait_for_timeout(self, timeout):
        """Wait for time out"""
        if not self._running:
            self.start(timeout)
        self._thrd.join()

    def _wait_for_timeout(self):
        cnt = 0
        self._running = True
        while not (self.condition or self._abort):
            time.sleep(self._dt)
            cnt += 1
            if cnt*self._dt > self._timeout:
                self._timeout_occurred = True
                self.event_timeout.raise_event()
                print('Timeout')
                break

        self._running = False

    def update(self, publisher):
        """When called, timout is canceled"""
        self.condition = True


class IUnitConverter(object):

    @staticmethod
    def convert(value, unit):
        raise NotImplementedError()


if __name__ == "__main__":

    print('--------Testing Timeout--------')
    # Create time out
    my_timeout = Timeout(dt=0.01)

    # Connect Event publisher and subscribers:
    sub = SimpleEventSubscriber(lambda _: print("Timed out"))
    pub = EventPublisher(data_type=bool)

    my_timeout.event_timeout.connect(sub)
    pub.connect(my_timeout)

    print('Test timeout (timeout expected):')
    my_timeout.start(.1)
    time.sleep(0.2)
    print('... done')

    print('Test condition change (no timeout expected):')
    my_timeout.start(.2)
    time.sleep(0.1)
    pub.raise_event(True)
    time.sleep(0.5)
    print('... done.')

    print('Test abort (no timeout expected)...')
    my_timeout.start(.1)
    my_timeout.abort()
    time.sleep(0.2)
    print('...done.')

    print('Test blocking call (timeout expected)...')
    my_timeout.wait_for_timeout(0.5)
    print('...done.')
