# author: Martijn Zeestraten
# data: 14-03-2018
import traitlets, logging, sys, os, asyncio
import nidaqmx.constants as nic
from nidaqmx.constants import EncoderType
import nidaqmx.stream_readers
import madernpytools.log as mlog
import madernpytools.constants as mcnst
from madernpytools.signal_handling import Signal, SignalDict, SignalList, ISignalProvider, ISignalKeyList, SignalKeyList
from madernpytools.tools.utilities import ListofDict
from madernpytools.backbone import *
from itertools import chain

import xml.etree.cElementTree as ET

import nidaqmx


logger = logging.getLogger(f'madernpytools.{__name__}')


class ClassFactory(IClassFactory):
    """ Factory class which allows to generate class instances from this module

    """

    @staticmethod
    def get(name):
        if name in sys.modules[__name__].__dict__:
            return eval(name)
        else:
            for mod in []:
                if name in getattr(mod, '__dict__'):
                    return getattr(mod, 'ClassFactory').get(name)
            raise TypeError('Could not find {}'.format(name))


def list_devices(verbose=False):
    """List available DAQ devices"""
    mysys = nidaqmx.system.System()

    if verbose:
        print('Devices found:')
        for d in mysys.devices:
            print('{0}\t: {1}'.format(d.name, d.product_category.name))

    return mysys.devices


class AbstractSensor(XMLSerializer):

    def __init__(self, device_name, channel_index, name='', unit='', sensor_type='', scale=1.0, **kwargs):
        self._device_name = device_name
        self._channel_index = channel_index
        self._name = name
        self._unit = unit
        self._sensor_type = sensor_type
        self._scale = scale
        XMLSerializer.__init__(self, **kwargs)

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def channel_index(self) -> int:
        return self._channel_index

    @property
    def type(self) -> str:
        return self._sensor_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def unit(self):
        return self._unit

    def add_to_ni_task(self, ni_task):
        raise  NotImplementedError()

    def __str__(self):
        return 'dev: {}, channel_index: {}, type: {}, name {}, unit: {}'.format(
            self.device_name, self.channel_index, self.type, self.name, self.unit)


class AnalogSensor(AbstractSensor):
    def __init__(self, **kwargs):
        AbstractSensor.__init__(self, **kwargs)

    def add_to_ni_task(self, ni_task):
        raise  NotImplementedError()


class CounterSensor(AbstractSensor):
    def __init__(self, **kwargs):
        AbstractSensor.__init__(self, **kwargs)

    def add_to_ni_task(self, ni_task):
        raise  NotImplementedError()


class SensorList(TraitsXMLSerializer, traitlets.TraitType):
    _ailist = traitlets.TraitType(default_value=[])
    _cilist = traitlets.TraitType(default_value=[])

    def __init__(self, ci_list: list=None,
                 ai_list: list=None, **kwargs):

        ci_list = [] if ci_list is None else ci_list
        ai_list = [] if ai_list is None else ai_list

        super().__init__(_cilist=ci_list, _ailist=ai_list,
                         var_names_mapping=[('ai_list', '_ailist'),
                                            ('ci_list', '_cilist')])

    def append(self, item):
        if isinstance(item, AnalogSensor):
            self._ailist.append(item)
        elif isinstance(item, CounterSensor):
            self._cilist.append(item)
        else:
            raise TypeError(f'Unknown type ({type(item)}), can only add AnalogSensor or CounterSensor.')

    def __getitem__(self, index):
        if index < len(self._ailist):
            return self._ailist[index]
        elif index < (len(self._ailist) + len(self._cilist)):
            return self._cilist[index - len(self._ailist)]

    def __setitem__(self, index, value):
        if isinstance(value, AnalogSensor) and index < len(self._ailist):
            self._ailist[index] = value
        elif isinstance(value, CounterSensor) and index < (len(self._ailist) + len(self._cilist)):
            self._cilist[index] = value
        elif isinstance(value, CounterSensor) and index < len(self._ailist):
            raise TypeError('Cannot set CounterSensor on AnalogSensor index.')
        elif isinstance(value, AnalogSensor) and index < (len(self._ailist) + len(self._cilist)):
            raise TypeError('Cannot set AnalogSensor on CounterSensor index.')
        elif index >= (len(self._ailist) + len(self._cilist)):
            raise IndexError('Index exceeds sensor length')

    def __iter__(self):
        return chain(self._ailist, self._cilist)

    def __len__(self):
        return len(self._ailist) + len(self._cilist)

    @property
    def sensor_names(self):
        return [s.name for s in self._ailist] + [s.name for s in self._cilist]


class PCBAccelerationSensor(AnalogSensor):

    def __init__(self, device_name, channel_index, name=''):
        """PCB accelerometer

        :param device_name  : task_name of the NI-DAQ module (e.g. MadernRDMod1)
        :param channel_index : index of the analog input on device module (e.g. 0)
        :param name: sensor task_name
        """
        if name=='':
            name = 'Acc{0}'.format(channel_index)

        AnalogSensor.__init__(self, device_name=device_name, channel_index=channel_index, name=name, unit='m/s2',
                              sensor_type='accelerometer')

    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """Add this sensor to given task

        """

        # Add channel:
        ni_task.ai_channels.add_ai_accel_chan('{0}/ai{1}'.format(self.device_name, self.channel_index),
                                              name_to_assign_to_channel=self.name,
                                              terminal_config=nic.TerminalConfiguration.PSEUDODIFFERENTIAL,
                                              min_val=-5.0, max_val=5.0,
                                              units=nic.AccelUnits.METERS_PER_SECOND_SQUARED,
                                              sensitivity=100,
                                              sensitivity_units=nic.AccelSensitivityUnits.M_VOLTS_PER_G,
                                              current_excit_source=nic.ExcitationSource.INTERNAL,
                                              current_excit_val=0.020,
                                              custom_scale_name=''
                                              )

class BruelAccelerationSensor(AnalogSensor):

    def __init__(self, device_name, channel_index, name=''):
        """PCB accelerometer

        :param device_name  : task_name of the NI-DAQ module (e.g. MadernRDMod1)
        :param channel_index: index of the analog input on device module (e.g. 0)
        :param name: Name to assign to sensor
        """
        if name=='':
            name = 'Acc{0}'.format(channel_index)

        AnalogSensor.__init__(self, device_name=device_name, channel_index=channel_index, name=name, unit='m/s2',
                              sensor_type='bruel accelerometer')


    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """Add sensor to task
        """

        ni_task.ai_channels.add_ai_accel_chan('{0}/ai{1}'.format(self.device_name, self.channel_index),
                                              name_to_assign_to_channel=self.name,
                                              terminal_config=nic.TerminalConfiguration.PSEUDODIFFERENTIAL,
                                              min_val=-5.0, max_val=5.0,
                                              units=nic.AccelUnits.METERS_PER_SECOND_SQUARED,
                                              sensitivity=90.21,
                                              sensitivity_units=nic.AccelSensitivityUnits.M_VOLTS_PER_G,
                                              current_excit_source=nic.ExcitationSource.INTERNAL,
                                              current_excit_val=0.004,
                                              custom_scale_name=''
                                              )


class PCBImpactHammerSensor(AnalogSensor):

    def __init__(self, device_name, channel_index, name='', sensitivity=0.2312):
        """PCB Impulse Hammer

        :param device_name  : task_name of the NI-DAQ module (e.g. MadernRDMod1)
        :param channel_index: index of the analog input on device module (e.g. 0)
        :param name: Name to assign to sensor
        :param sensitivity: [mV/N] See hammer specification or calibration certificate.

        """
        if name == '':
            name = 'impact_hammer_{0}'.format(channel_index)

        self._sensitivity = sensitivity

        # ----
        AnalogSensor.__init__(self, device_name=device_name, channel_index=channel_index, name=name, unit='m/s2',
                              sensor_type='PCB Impact Hammer')

        # Correct var_names mapping
        self._var_names_mapping = list(zip(['sensitivity', 'device_name', 'channel_index', 'task_name'],
                                           self.__dict__.keys()))

    @property
    def sensitivity(self):
        """ Sensor sensitivity

        """
        return self._sensitivity

    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """ Add sensor to given task

        """

        # Add task to channel
        ni_task.ai_channels.add_ai_force_iepe_chan('{0}/ai{1}'.format(self.device_name, self.channel_index),
                                                   name_to_assign_to_channel=self.name,
                                                   terminal_config=nic.TerminalConfiguration.PSEUDODIFFERENTIAL,
                                                   min_val=-5.0, max_val=5.0,
                                                   units=nic.ForceUnits.NEWTONS,
                                                   sensitivity=self.sensitivity,
                                                   sensitivity_units=nic.ForceIEPESensorSensitivityUnits.M_VOLTS_PER_NEWTON,
                                                   current_excit_source=nic.ExcitationSource.INTERNAL,
                                                   current_excit_val=0.020,
                                                   custom_scale_name=''
                                                   )


class MEcapaNCDTSensor(AnalogSensor):

    def __init__(self, device_name, channel_index, name='', scale=100.0, unit='mu'):
        """Micro-Epsilon capacitive sensor capaNCDT

        :param device_name  : task_name of the NI-DAQ module (e.g. MadernRDMod1)
        :param channel_index: index of the analog input on device module (e.g. 0)
        :param scale:  (micron/volt) Conversion between sensor voltage and distance (default 100 micron/Volt)
        :param name: Signal task_name
        :returns: signal associated to the sensor
        """
        if name == '':
            name = 'capaNCDT{0}'.format(channel_index)
        names = ['device_name', 'channel_index', 'name', 'scale', 'unit']
        AnalogSensor.__init__(self, device_name=device_name, channel_index=channel_index, name=name, unit=unit,
                              sensor_type='ME capaNCDT', scale=scale,
                              var_names_mapping=list(zip(names, [f'_{name}' for name in names])))

    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """ Add sensor to given task

        """

        ni_task.ai_channels.add_ai_voltage_chan(physical_channel='{0}/ai{1}'.format(self.device_name, self.channel_index),
                                                name_to_assign_to_channel='cap{0}'.format(self.channel_index),
                                                terminal_config=nic.TerminalConfiguration.DIFFERENTIAL,
                                                min_val=-10.0,
                                                max_val= 10.0,
                                                units=nic.VoltageUnits.VOLTS,
                                                custom_scale_name=''
                                                )


class VoltageSensor(AnalogSensor):

    def __init__(self, device_name, channel_index, name='', min_val=-10, max_val=10.0):
        """Add channel for generic voltage measurement in Volts.

        :param device_name: Device task_name on which the channel is located
        :param channel_index: Analog input on which the signal is connected
        :param min_val: Maximum expected current [V]
        :param max_val: Minimum expected current [V]
        :param name: Name of the signal (task_name is used to label this signal)
        """
        if name == '':
            name = 'Voltage{0}'.format(channel_index)

        self._min_val = min_val
        self._max_val = max_val

        names = ['device_name', 'channel_index', 'name', 'min_val', 'max_val']
        AnalogSensor.__init__(self, device_name=device_name, channel_index=channel_index, name=name, unit='Volt',
                              sensor_type='Voltage',
                              var_names_mapping=list(zip(names, [f'_{name}' for name in names]))
                              )

        # Correct var_names mapping
        self._var_names_mapping = list(zip(['min_val', 'max_val', 'device_name', 'channel_index', 'name',
                                            ],
                                           self.__dict__.keys()))

    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """Add sensor to task
        """

        """Add channel for generic voltage measurement in Volts.

        :param dev_name: Device task_name on which the channel is located
        :param ai_index: Analog input on which the signal is connected
        :param min_val: Maximum expected current [V]
        :param max_val: Minimum expected current [V]
        :param task_name: Name of the signal (task_name is used to label this signal)
        """
        ni_task.ai_channels.add_ai_voltage_chan(physical_channel='{0}/ai{1}'.format(self.device_name, self.channel_index),
                                                name_to_assign_to_channel=self.name,
                                                terminal_config=nic.TerminalConfiguration.DIFFERENTIAL,
                                                min_val= self._min_val,
                                                max_val= self._max_val,
                                                units=nic.VoltageUnits.VOLTS,
                                                custom_scale_name=''
                                                )


class CurrentSensor(AnalogSensor):

    def __init__(self, device_name, channel_index, name='', min_val=-0.02, max_val=0.02):
        """Add channel for generic current measurement in Ampere

        :param device_name: Device task_name on which the channel is located
        :param channel_index: Analog input on which the signal is connected
        :param min_val: Maximum expected current [A]
        :param max_val: Minimum expected current [A]
        :param name: Name of the signal (task_name is used to label this signal)
        """
        if name == '':
            name = 'Current{0}'.format(channel_index)

        self._min_val = min_val
        self._max_val = max_val

        AnalogSensor.__init__(self, device_name=device_name, channel_index=channel_index, name=name, unit='A',
                              sensor_type='Current')

        # Correct var_names mapping
        self._var_names_mapping = list(zip(['min_val', 'max_val', 'device_name', 'channel_index', 'name',
                                            ],
                                           self.__dict__.keys()))

    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """Add sensor to task
        """

        ni_task.ai_channels.add_ai_voltage_chan(physical_channel='{0}/ai{1}'.format(self.device_name, self.channel_index),
                                                name_to_assign_to_channel=self.device_name,
                                                terminal_config=nic.TerminalConfiguration.DIFFERENTIAL,
                                                min_val=self._min_val,
                                                max_val=self._max_val,
                                                units=nic.CurrentUnits.AMPS,
                                                custom_scale_name=''
                                                )


class Thermocouple(AnalogSensor):

    def __init__(self, device_name, channel_index, name='', min_val=0.0, max_val=100,
                 thermocouple_type: nic.ThermocoupleType=nic.ThermocoupleType.J):
        """Add channel for generic current measurement in Ampere

        :param device_name: Device task_name on which the channel is located
        :param channel_index: Analog input on which the signal is connected
        :param min_val: Minimum expected temperature [degC]
        :param max_val: Maximum expected temperature [degC]
        :param name: Name of the signal (task_name is used to label this signal)
        """


        if name == '':
            name = 'TC{0}'.format(channel_index)

        self._min_val = min_val
        self._max_val = max_val

        AnalogSensor.__init__(self, device_name=device_name, channel_index=channel_index, name=name, unit='deg_c',
                              sensor_type='Thermocouple')

        # Correct var_names mapping
        self._var_names_mapping = list(zip(['min_val', 'max_val', 'device_name', 'channel_index', 'name',
                                            ],
                                           self.__dict__.keys()))

        self._type = thermocouple_type # TODO: We currenlty ignore this with serialization

    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """Add sensor to task
        """
        ni_task.ai_channels.add_ai_thrmcpl_chan(
            physical_channel='{0}/ai{1}'.format(self.device_name, self.channel_index),
            name_to_assign_to_channel="",
            min_val=self._min_val,
            max_val=self._max_val,
            units=nic.TemperatureUnits.DEG_C,
            thermocouple_type=self._type,
            cjc_source=nic.CJCSource.CONSTANT_USER_VALUE,
            cjc_val=25.0,
            cjc_channel="")


class LinearEncoder(CounterSensor):

    def __init__(self, device_name, counter_index, decoding_type, name=''):
        """Linear encoder signal.

        Note encoder signal is digital, and cannot be combined with analog tasks

        :param name: Name to assign to encoder channel
        :param device_name: Module to which encoder is connected
        :param counter_index: Index of counter to use
        :param decoding_type: Encoder type
        :return:
        """
        if name == '':
            name = 'encoder{0}'.format(counter_index)

        self._decoding_type = decoding_type

        CounterSensor.__init__(self, device_name=device_name, channel_index=counter_index, name=name, unit='A',
                               sensor_type='LinearEncoder')

        # Correct var_names mapping
        self._var_names_mapping = list(zip(['decodint_type', 'device_name', 'channel_index', 'name' ],
                                           self.__dict__.keys()))

    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """Add sensor to task
        """
        ni_task.ci_channels.add_ci_lin_encoder_chan(counter='{0}/ctr{1}'.format(self.device_name, self.channel_index),
                                                    name_to_assign_to_channel=self.name,
                                                    decoding_type=self._decoding_type)


class AngularX4Encoder(CounterSensor):

    def __init__(self, device_name, counter_index, name='', pulses_per_rev=1024):
        """ Add angular encoder signal.

        Note encoder signal is digital, and cannot be combined with analog tasks

        :param name: Name to assign to encoder channel
        :param device_name: Module to which encoder is connected
        :param counter_index: Index of counter to use
        :param pulses_per_rev: Number of pulses in one encoder revolution(see nidaqmx doc)
        """
        if name == '':
            name = 'encoder{0}'.format(counter_index)

        self._pulses_per_rev = pulses_per_rev

        CounterSensor.__init__(self, device_name=device_name, channel_index=counter_index, name=name, unit='rad',
                               sensor_type='AngularEncoder')

        self._var_names_mapping = list(zip(['pulses_per_rev', 'device_name', 'counter_index', 'name'],
                                           self.__dict__.keys()))

    def add_to_ni_task(self, ni_task: nidaqmx.Task):
        """Add sensor to task
        """

        # Fixed settings

        # Create channel
        ni_task.ci_channels.add_ci_ang_encoder_chan(counter='{0}/ctr{1}'.format(self.device_name, self.channel_index),
                                                    name_to_assign_to_channel=self.name,
                                                    zidx_enable=True,
                                                    zidx_phase=nic.EncoderZIndexPhase.AHIGH_BHIGH,
                                                    units=nic.AngleUnits.RADIANS,
                                                    pulses_per_rev=self._pulses_per_rev,
                                                    initial_angle=0.0,
                                                    zidx_val=0.0,
                                                    decoding_type=nic.EncoderType.X_4,
                                                    )


class AbstractMadernSensors(DataSubscriber):
    """Interface class for Madern Sensors"""

    def __init__(self):
        DataSubscriber.__init__(self)

    def __getitem__(self, key):
        raise NotImplementedError()

    def cb_new_data(self, publisher):
        raise NotImplementedError()

    @property
    def signals(self):
        raise NotImplementedError()

    def get_signal_index(self, signal: Signal) -> int:
        raise NotImplementedError()


class CISensors(AbstractMadernSensors, DataPublisher):

    def __init__(self, ni_task: nidaqmx.Task):
        AbstractMadernSensors.__init__(self)
        DataPublisher.__init__(self)
        self._task = ni_task
        self._signals = SignalDict()
        self.connect(self._signals)

        self.event_signal_added = EventPublisher(data_type=Signal)

    def __getitem__(self, key):
        return self._task[key]

    def keys(self):
        return self._signals.keys()

    def __len__(self):
        return len(self._signals)

    def _add_signal(self, name, unit, module, channel_index):
        if not (name in self._signals.keys()):
            new_signal = Signal(name=name, unit=unit, data_index=len(self._signals.keys()),
                                module=module, channel_index=channel_index)
            self._signals[name] = new_signal   # Add to list
            self.event_signal_added.raise_event(new_signal)
            return new_signal
        else:
            raise RuntimeError("Signal with task_name <{0}>, and unit <{1}>, already exits".format(name, unit))

    def cb_new_data(self, publisher):
        """Handles new data"""
        self.raise_event(publisher.get_data())

    def add_linear_encoder(self, name, counter_module, counter_index, decoding_type=nic.EncoderType.X_4, **kwargs):
        """ Add linear encoder signal.

        Note encoder signal is digital, and cannot be combined with analog tasks

        :param name: Name to assign to encoder channel
        :param counter_module: Module to which encoder is connected
        :param counter_index: Index of counter to use
        :param decoding_type: Encoder type
        :param kwargs: Optional arguments (see nidaqmx for remaining arguments)
        :return:
        """
        self._task.ci_channels.add_ci_lin_encoder_chan(counter='{0}/ctr{1}'.format(counter_module, counter_index),
                                                       name_to_assign_to_channel=name,
                                                       decoding_type=decoding_type,
                                                       **kwargs)

        return self._add_signal(name=name, unit='m/min', module=counter_module, channel_index=counter_index)

    def add_angular_encoder(self, name, module_name, counter_index, decoding_type=nic.EncoderType.X_4,
                            zidx_enable=True, zidx_phase=nic.EncoderZIndexPhase.AHIGH_BHIGH, zidx_val=0.0,
                            units=nic.AngleUnits.RADIANS, pulses_per_rev=1024, initial_angle=0.0):
        """ Add linear encoder signal.

        Note encoder signal is digital, and cannot be combined with analog tasks

        :param name: Name to assign to encoder channel
        :param module_name: Module to which encoder is connected
        :param counter_index: Index of counter to use
        :param decoding_type: Encoder type
        :param zidx_enable: Enable reset (Z signal)
        :param zidx_phase: (see nidaqmx doc)
        :param zidx_val: (see nidaqmx doc)
        :param units: (see nidaqmx doc)
        :param pulses_per_rev: Number of pulses in one encoder revolution(see nidaqmx doc)
        :param initial_angle: Initial encoder angle (see nidaqmx doc)
        :return Signal: encoder corresponding to encoder sensor
        """
        self._task.ci_channels.add_ci_ang_encoder_chan(counter='{0}/ctr{1}'.format(module_name, counter_index),
                                                       zidx_enable=zidx_enable, zidx_phase=zidx_phase, zidx_val=zidx_val,
                                                       units=units, pulses_per_rev=pulses_per_rev,
                                                       initial_angle=initial_angle,
                                                       name_to_assign_to_channel=name,
                                                       decoding_type=decoding_type)

        return self._add_signal(name=name, unit='radian', module=module_name, channel_index=counter_index)

    def add_time(self):
        """Add time to signals"""
        return self._add_signal(name='time', unit='s', module='-', channel_index=-1)

    @property
    def signals(self):
        """Dictionary of signals"""
        return self._signals

    def get_signal_index(self, signal: Signal):
        return self._signals[signal.name].data_index


class AISensors(AbstractMadernSensors, DataPublisher):

    def __init__(self, task):
        AbstractMadernSensors.__init__(self)
        DataPublisher.__init__(self)
        self._task = task
        self._signals = SignalDict()
        self.connect(self._signals)

        self.event_signal_added = EventPublisher(data_type=Signal)

    def __getitem__(self, key):
        return self._task[key]

    def keys(self):
        return self._signals.keys()

    def __len__(self):
        return len(self._signals)

    def _add_signal(self, name, unit, module, channel_index, scale=1.0):
        if not (name in self._signals.keys()):
            new_signal = Signal(name=name, unit=unit, data_index=len(self._signals.keys()),
                                module=module, channel_index=channel_index, scale=scale)
            self._signals[name] = new_signal   # Add to list
            self.event_signal_added.raise_event(new_signal)
            return new_signal
        else:
            raise RuntimeError("Signal with task_name <{0}>, and unit <{1}>, already exits".format(name, unit))

    def add_sensor(self, sensor:AbstractSensor):

        if sensor.type == 'acceleration':
            self.add_pcb_accelerometer(dev_name=sensor.device_name, ai_index=sensor.channel_index,name=sensor.name)
        elif sensor.type=='voltage':
            self.add_voltage_input(dev_name=sensor.device_name, ai_index=sensor.channel_index,name=sensor.name)
        elif sensor.type=='acceleration_bruel':
            self.add_voltage_input(dev_name=sensor.device_name, ai_index=sensor.channel_index,name=sensor.name)


    def cb_new_data(self, publisher):
        """Handles new data"""
        self.raise_event(publisher.get_data())

    def add_pcb_accelerometer(self, dev_name, ai_index, name=''):
        """ Add PCB accelerometer

        :param dev_name  : task_name of the NI-DAQ module (e.g. MadernRDMod1)
        :param ai_index : index of the analog input on device module (e.g. 0)
        :returns: signal associated to the sensor
        """

        if name=='':
            name = 'Acc{0}'.format(ai_index)

        self._task.ai_channels.add_ai_accel_chan('{0}/ai{1}'.format(dev_name, ai_index),
                                                 name_to_assign_to_channel=name,
                                                 terminal_config=nic.TerminalConfiguration.PSEUDODIFFERENTIAL,
                                                 min_val=-5.0, max_val=5.0,
                                                 units=nic.AccelUnits.METERS_PER_SECOND_SQUARED,
                                                 sensitivity=100,
                                                 sensitivity_units=nic.AccelSensitivityUnits.M_VOLTS_PER_G,
                                                 current_excit_source=nic.ExcitationSource.INTERNAL,
                                                 current_excit_val=0.020,
                                                 custom_scale_name=''
                                                 )

        return self._add_signal(name=name, unit='m/s^2', module=dev_name, channel_index=ai_index)

    def add_bruel_accelerometer(self, dev_name, ai_index, name=''):
        """ Add PCB accelerometer

        :param dev_name  : task_name of the NI-DAQ module (e.g. MadernRDMod1)
        :param ai_index : index of the analog input on device module (e.g. 0)
        :returns: signal associated to the sensor
        """

        if name=='':
            name = 'Acc{0}'.format(ai_index)

        self._task.ai_channels.add_ai_accel_chan('{0}/ai{1}'.format(dev_name, ai_index),
                                                 name_to_assign_to_channel=name,
                                                 terminal_config=nic.TerminalConfiguration.PSEUDODIFFERENTIAL,
                                                 min_val=-5.0, max_val=5.0,
                                                 units=nic.AccelUnits.METERS_PER_SECOND_SQUARED,
                                                 sensitivity=90.21,
                                                 sensitivity_units=nic.AccelSensitivityUnits.M_VOLTS_PER_G,
                                                 current_excit_source=nic.ExcitationSource.INTERNAL,
                                                 current_excit_val=0.004,
                                                 custom_scale_name=''
                                                 )

        return self._add_signal(name=name, unit='m/s^2', module=dev_name, channel_index=ai_index)

    def add_pcb_impulsehammer(self, dev_name, ai_index, sensitivity=0.2321, name = ''):
        """ Add PCB Impulse Hammer

        :param dev_name  : task_name of the NI-DAQ module (e.g. MadernRDMod1)
        :param ai_index : index of the analog input on device module (e.g. 0)
        :param sensitivity: [mV/N] See hammer specification or calibration certificate.
        :returns: signal associated to the sensor
        """
        if name=='':
            name = 'ImpactHammer'.format(ai_index)
        self._task.ai_channels.add_ai_force_iepe_chan('{0}/ai{1}'.format(dev_name, ai_index),
                                                      name_to_assign_to_channel=name,
                                                      terminal_config=nic.TerminalConfiguration.PSEUDODIFFERENTIAL,
                                                      min_val=-5.0, max_val=5.0,
                                                      units=nic.ForceUnits.NEWTONS,
                                                      sensitivity=sensitivity,
                                                      sensitivity_units=nic.ForceIEPESensorSensitivityUnits.M_VOLTS_PER_NEWTON,
                                                      current_excit_source=nic.ExcitationSource.INTERNAL,
                                                      current_excit_val=0.020,
                                                      custom_scale_name=''
                                                      )

        return self._add_signal(name=name, unit='N', module=dev_name, channel_index=ai_index)

    def add_microepsilon_capacitive(self, dev_name, ai_index, name=''):
        """ Micro-Epsilon capacitive sensor

        :param dev_name  : task_name of the NI-DAQ module (e.g. MadernRDMod1)
        :param ai_index : index of the analog input on device module (e.g. 0)
        :param name: Signal task_name
        :returns: signal associated to the sensor
        """

        if name=='':
            name = 'cap{0}'.format(ai_index)

        self._task.ai_channels.add_ai_voltage_chan(physical_channel='{0}/ai{1}'.format(dev_name, ai_index),
                                                   name_to_assign_to_channel='cap{0}'.format(ai_index),
                                                   terminal_config=nic.TerminalConfiguration.DIFFERENTIAL,
                                                   min_val=-10.0,
                                                   max_val= 10.0,
                                                   units=nic.VoltageUnits.VOLTS,
                                                   custom_scale_name=''
                                                   )

        return self._add_signal(name=name, unit='\mu m', module=dev_name, channel_index=ai_index, scale=100.0)

    def add_voltage_input(self, dev_name, ai_index, name='', min_val=-10.0, max_val=10.0):
        """Add channel for generic voltage measurement in Volts.

        :param dev_name: Device task_name on which the channel is located
        :param ai_index: Analog input on which the signal is connected
        :param min_val: Maximum expected current [V]
        :param max_val: Minimum expected current [V]
        :param name: Name of the signal (task_name is used to label this signal)
        """

        self._task.ai_channels.add_ai_voltage_chan(physical_channel='{0}/ai{1}'.format(dev_name, ai_index),
                                                   name_to_assign_to_channel=dev_name,
                                                   terminal_config=nic.TerminalConfiguration.DIFFERENTIAL,
                                                   min_val= min_val,
                                                   max_val= max_val,
                                                   units=nic.VoltageUnits.VOLTS,
                                                   custom_scale_name=''
                                                   )
        return self._add_signal(name=name, unit='Volts', module=dev_name, channel_index=ai_index)

    def add_current_input(self, dev_name, ai_index, name='', min_val=-0.02, max_val=0.02):
        """Add channel for generic current measurement in Ampere

        :param dev_name: Device task_name on which the channel is located
        :param ai_index: Analog input on which the signal is connected
        :param min_val: Maximum expected current [A]
        :param max_val: Minimum expected current [A]
        :param name: Name of the signal (task_name is used to label this signal)
        """
        self._task.ai_channels.add_ai_voltage_chan(physical_channel='{0}/ai{1}'.format(dev_name, ai_index),
                                                   name_to_assign_to_channel=dev_name,
                                                   terminal_config=nic.TerminalConfiguration.DIFFERENTIAL,
                                                   min_val=min_val,
                                                   max_val=max_val,
                                                   units=nic.CurrentUnits.AMPS,
                                                   custom_scale_name=''
                                                   )
        return self._add_signal(name=name, unit='Amps', module=dev_name, channel_index=ai_index)

    def add_thermocouple(self, dev_name, ai_index, name='', min_val=0.0, max_val=100):
        """Add channel for generic current measurement in Ampere

        :param dev_name: Device task_name on which the channel is located
        :param ai_index: Analog input on which the signal is connected
        :param min_val: Maximum expected current [A]
        :param max_val: Minimum expected current [A]
        :param name: Name of the signal (task_name is used to label this signal)
        """
        self._task.ai_channels.add_ai_thrmcpl_chan(
            physical_channel='{0}/ai{1}'.format(dev_name, ai_index),
            name_to_assign_to_channel="",
            min_val=min_val,
            max_val=max_val,
            units=nic.TemperatureUnits.DEG_C,
            thermocouple_type=nic.ThermocoupleType.J,
            cjc_source=nic.CJCSource.CONSTANT_USER_VALUE,
            cjc_val=25.0,
            cjc_channel="")

        return self._add_signal(name=name, unit='deg_c', module=dev_name, channel_index=ai_index)

    def add_time(self):
        """Add time to signals"""
        return self._add_signal(name='time', unit='s', module='-', channel_index=-1)

    @property
    def signals(self):
        """Dictionary of signals"""
        return self._signals


class TaskConfiguration(TraitsXMLSerializer, traitlets.TraitType):
    sampling_rate = traitlets.CInt(default_value=10240)
    buffer_size = traitlets.CInt(default_value=200000)
    clock_source = traitlets.CUnicode(default_value='')
    chassis_name = traitlets.CUnicode(default_value='SimMadernRD')
    task_name = traitlets.CUnicode(default_value='MadernTask')

    def __init__(self, sampling_rate: int=10240, buffer_size: int=200000, clock_source: str = None,
                 chassis_name: str = 'SimMadernRD', task_name: str ='MadernTask'):
        """
        :param sampling_rate : sampling frequency in Hertz
        :param buffer_size: Measurement duration in seconds
        :param clock_source: Name of clock source (required for digital tasks, e.g. encoder)
        """
        clock_source = ArgumentVerifier(str, 'None').verify(clock_source)

        names = ['sampling_rate', 'buffer_size', 'clock_source', 'chassis_name', 'task_name']
        super().__init__(sampling_rate=sampling_rate,
                         buffer_size=buffer_size,
                         clock_source=clock_source,
                         chassis_name=chassis_name,
                         task_name=task_name,
                         var_names_mapping=list(zip(names, names))
                         )


class AbstractMadernTask(ISignalProvider, traitlets.HasTraits, traitlets.TraitType):

    output_data = traitlets.TraitType()
    configuration = TaskConfiguration()
    #signals = traitlets.Dict(default_value={})
    sensors = SensorList()

    def start(self):
        """Start ni_task"""
        raise NotImplementedError()

    @traitlets.default('sensors')
    def _default_sensors(self):
        return SensorList()

    @traitlets.default('output_data')
    def _default_sensors(self):
        return ListofDict(items=[])

    @traitlets.default('configuration')
    def _default_configuration(self):
        return TaskConfiguration()

    @traitlets.observe('configuration')
    def _configuration_change(self, change):
        raise NotImplementedError()

    def stop(self):
        """Stop ni_task"""
        raise NotImplementedError()

    def close(self):
        """Close ni_task"""
        raise NotImplementedError()

    def reset(self):
        """Reset the ni_task"""
        raise NotImplementedError()

    @property
    def required_input_keys(self) -> ISignalKeyList:
        """Signals required """
        raise NotImplementedError()

    @property
    def added_keys(self) -> ISignalKeyList:
        """Signals """
        raise NotImplementedError()


class MadernTask(TraitsXMLSerializer, AbstractMadernTask):

    def __init__(self, configuration: TaskConfiguration = None, sensors: SensorList=None, **kwargs):
        """
        :param configuration: configuration
        :param sensors: List of _sensors
        """

        sensors = ArgumentVerifier(SensorList, SensorList()).verify(sensors)

        # Initializaton flags
        self._ai_task = None
        self._ci_task = None
        self._ai_configured = False
        self._ci_configured = False

        self._queue_task = None #asyncio.Task()
        self._queue = asyncio.Queue()

        # Variables for device reading:
        self._last_t = 0  # Time keeper

        var_names_mapping = [('configuration', 'configuration'), ('sensors', 'sensors')]
        super().__init__(var_names_mapping=var_names_mapping,
                         sensors=sensors,
                         configuration=configuration)

    @traitlets.observe('configuration')
    def _configuration_change(self, change):
        if isinstance(change.new, TaskConfiguration):
            # Reserve network device
            nidaqmx.system.Device(self.configuration.chassis_name).reserve_network_device(True)

            # Close existing tasks:
            for task in [self._ai_task, self._ci_task]:
                if isinstance(task, nidaqmx.Task):
                    task.close()

            # Create new tasks:
            self._ai_task = nidaqmx.Task(new_task_name='{0}_ai'.format(self.configuration.task_name))
            self._ci_task = nidaqmx.Task(new_task_name='{0}_ci'.format(self.configuration.task_name))

            # Set sensors sensors:
            for s in self.sensors:
                self._add_sensor_to_task(s)

            # Apply configuration
            self._configure_tasks()

            # Create queue
            if self._queue_task is None:
                self._queue_task = asyncio.create_task(self._queue_worker())

    def _configure_tasks(self):
        raise NotImplementedError()

    @traitlets.observe('sensors')
    def _sensors_change(self, change):
        if self._ai_configured:
            for sensor in self.sensors:
                self._add_sensor_to_task(sensor)

    def add_sensor(self, sensor: AbstractSensor):
        # Add to task:
        if self._ai_configured:
            self._add_sensor_to_task(sensor)

        # Add to internal dict:
        self.sensors.append(sensor)

    def _add_sensor_to_task(self, sensor):
        if isinstance(sensor, AnalogSensor):
            # Add sensor to task:
            sensor.add_to_ni_task(self._ai_task)

        elif isinstance(sensor, CounterSensor):
            sensor.add_to_ni_task(self._ci_task)

    @property
    def ai_task(self):
        return self._ai_task

    @property
    def ci_task(self):
        return self._ci_task

    @property
    def sampling_rate(self):
        if self._ai_configured:
            return int(self.ai_task.timing.samp_clk_rate)
        else:
            return self.configuration.sampling_rate

    @property
    def ci_sensors(self):
        return [s for s in self.sensors if isinstance(s, CounterSensor)]

    @property
    def ai_sensors(self):
        return [s for s in self.sensors if isinstance(s, AnalogSensor)]

    def _read_device(self, *args):
        # Check only when running:
        # Read data:
        new_ai_data = np.array(self.ai_task.read(nic.READ_ALL_AVAILABLE)).T

        if new_ai_data.ndim == 1 and len(self.ai_sensors) == 1:
            # Data contains a single sensor (time is added next, but taken into self._sensors
            new_ai_data = new_ai_data[:, None]

        # Scale measured data:
        for i, s in enumerate(self.ai_sensors):
            new_ai_data[:, i] *= s.scale

        if self._ci_configured:
            new_ci_data = np.array(self.ci_task.read(len(new_ai_data))).T
            if new_ci_data.ndim == 1:
                new_ci_data = new_ci_data[:, None]
            new_data = np.hstack([new_ai_data, new_ci_data])

        else:
            new_data = new_ai_data

        if new_data.shape[0] > 0:
            # Add time steps:
            t = (np.arange(new_data.shape[0]) + 1)/self.sampling_rate + self._last_t
            new_data = np.hstack([t[:, None], new_data])

            # Update time
            self._last_t = t[-1]

            # Notify subscribers:
            logger.info(f'Putting {len(new_data)} data to queue (length: {self._queue.qsize()})')

            self._queue.put_nowait(self._set_output(new_data))
        else:
            pass

        return 0

    async def _queue_worker(self):
        logger.info('Starting queue worker...')
        while True:
            # Get new task from queue;
            #logger.info('Awaiting task...')
            # TODO: Find bug in this:
            # For some reason, await self._queue.get() does not work in combination with a QT-gui
            # As a result, I implemented the following if-else to

            if self._queue.qsize() > 0:
                task_routine = self._queue.get_nowait()
                #task_routine = await self._queue.get()
                logger.info('Got new task...')
                # Run task:
                await asyncio.create_task(task_routine)

                # Notify processed:
                self._queue.task_done()
            else:
                await asyncio.sleep(0.02)
            #task = await self._queue.get()

        logger.info('Finished...')

    async def _set_output(self, data):
        logger.info('Executing set output')
        new_dict = ListofDict([dict(zip(self.added_keys, line)) for line in data])
        self.output_data = new_dict

    def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def reset(self):
        self._last_t = 0

    def close(self):
        self.ai_task.close()
        self.ci_task.close()

    @property
    def required_input_keys(self) -> ISignalKeyList:
        return SignalKeyList([])

    @property
    def added_keys(self) -> ISignalKeyList:
        return ['time'] + self.sensors.sensor_names


class ContinuousMadernTask(MadernTask, traitlets.HasTraits):

    def __init__(self, *args, configuration: TaskConfiguration = None, sensors: SensorList= None, **kwargs):
        """ Initialize Measurement ni_task and reserve measurement device interface
        :param chassis_name: Name of NI-DAQ chassis
        """
        # Initialize concrete signals:
        var_names_mapping = [('sensors', 'sensors'), ('configuration', 'configuration')]

        if isinstance(configuration, TaskConfiguration):
            super().__init__(*args, sensors=sensors,
                             configuration=configuration,
                             var_names_mapping=var_names_mapping, **kwargs)
        else:
            super().__init__(*args, sensors=sensors,
                             var_names_mapping=var_names_mapping, **kwargs)

    def _configure_tasks(self):
        """  Madern DAQ Task

        """
        configuration = self.configuration

        clock_source = configuration.clock_source
        sampling_rate = configuration.sampling_rate
        buffer_size = configuration.buffer_size

        Ntimes = int(buffer_size/sampling_rate)

        logger.info(f'Setting sampling rate to {sampling_rate}')

        clock_source = ArgumentVerifier(str, '').verify(clock_source)

        # XML serialization could cause clock_source to become 'None', which corresponds to '', solve this bug here:
        if not isinstance(clock_source, str) or clock_source == 'None':
            clock_source = ''

        for i in range(2):
            self.ai_task.timing.cfg_samp_clk_timing(sampling_rate,
                                                    active_edge=nic.Edge.RISING,
                                                    sample_mode=nic.AcquisitionType.CONTINUOUS,
                                                    samps_per_chan=buffer_size
                                                    )
            sampling_rate = int(self.ai_task.timing.samp_clk_rate)
            buffer_size = sampling_rate * Ntimes
        self.ai_task.control(nic.TaskMode.TASK_COMMIT)
        self._ai_configured = True

        if len(self.ci_sensors) > 0:
            logger.info(f'Setting ci_task clock_source to {self.ai_task.timing.samp_clk_src}')

            # Set default clock_source to chassis ai SampleClock
            if clock_source == '':
                clock_source = f'/{self.configuration.chassis_name}/ai/SampleClock'

            self.ci_task.timing.cfg_samp_clk_timing(sampling_rate,
                                                    source=clock_source, #self.ai_task.timing.samp_clk_src,
                                                    sample_mode=nic.AcquisitionType.CONTINUOUS,
                                                    active_edge=nic.Edge.RISING,
                                                    samps_per_chan=buffer_size)
            self.ci_task.control(nic.TaskMode.TASK_COMMIT)
            self._ci_configured = True

        # Update configuration:
        logger.info(f'Updating configuration {self.ai_task.timing.samp_clk_rate}')
        self.configuration.sampling_rate = sampling_rate #self.ai_task.timing.samp_clk_rate
        self.configuration.buffer_size = buffer_size

        self._read_every = sampling_rate

        # Update configuration settings:
        clock_source = 'None' if clock_source == '' else clock_source

    def start(self):
        """Start ni_task execution"""
        if not self._ai_configured and isinstance(self.configuration, TaskConfiguration):
            self.configure(self.configuration)
        elif not self._ai_configured:
            raise RuntimeError('Task not configured, please configure first')

        if self._ai_configured:
            # Start handling of data, if not yet started
            if self._queue_task is None:
                self._queue_task = asyncio.create_task(self._queue_worker())
            elif self._queue_task.cancelled() or self._queue_task.done():
                self._queue_task = asyncio.create_task(self._queue_worker())

            # Clear autosave path:
            self.reset()
            self.ai_task.register_every_n_samples_acquired_into_buffer_event(self._read_every, self._read_device)
            self.ai_task.start()

            if self._ci_configured:
                self.ci_task.start()
        else:
            raise RuntimeError('Task is not configured, please configure first')
        return mcnst.Errors.no_error

    def stop(self):
        """Stop ni_task execution"""
        self._queue_task.cancel()
        self.ai_task.stop()
        if self._ci_configured:
            self.ci_task.stop()
        self.ai_task.register_every_n_samples_acquired_into_buffer_event(10, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def wait_until_done(self, timeout=10):
        self.ai_task.wait_until_done(timeout=timeout)


class DiscreteMadernTask(MadernTask):

    def __init__(self, sensors: list, configuration: TaskConfiguration):
        """ Initialize Measurement ni_task and reserve measurement device interface
        :param chassis_name: Name of NI-DAQ chassis
        """

        super().__init__(configuration=configuration, sensors=sensors)

        self._filename = '{0}.csv'.format(self.configuration.task_name)

    def _configure_tasks(self):
        """  Madern DAQ Task

        Parameters:
        :param sampling_rate : sampling frequency in Hertz
        :param buffer_size   : Number of samples in buffer
        :param source: Name of the clock source (required for digital tasks)
        """
        buffer_size = self.configuration.buffer_size
        sampling_rate = self.configuration.sampling_rate
        clock_source = self.configuration.clock_source

        logger.info(f'Setting sampling rate to {sampling_rate}')
        logger.info(f'Setting buffersize to {buffer_size}')
        N = buffer_size
        self.ai_task.timing.cfg_samp_clk_timing(sampling_rate,
                                                source=clock_source,
                                                active_edge=nic.Edge.RISING,
                                                sample_mode=nic.AcquisitionType.FINITE,
                                                samps_per_chan=int(N)
                                                )

        logger.info(f'Received clock rate {self.ai_task.timing.samp_clk_rate}')

        self._ai_configured = True
        if len(self.ci_task.channel_names) > 0:

            # Set default clock_source to chassis ai SampleClock
            if clock_source == '':
                clock_source = f'/{self.configuration.chassis_name}/ai/SampleClock'

            self.ci_task.timing.cfg_samp_clk_timing(self.ai_task.timing.samp_clk_rate,
                                                    source=clock_source,
                                                    sample_mode=nic.AcquisitionType.FINITE,
                                                    samps_per_chan=int(N))
            self._ci_configured = True

        # Update configuration settings:
        self._configuration = TaskConfiguration(sampling_rate, buffer_size, clock_source)

    def start(self):
        """Start ni_task execution"""
        if self._ai_configured:
            self.ai_task.register_done_event(self._cb_done_event)
            self.ai_task.start()
            if self._ci_configured:
                self.ci_task.start()
        else:
            raise RuntimeError("Task is not configured")
        return mcnst.Errors.no_error

    def stop(self):
        """Stop ni_task execution"""
        self.ai_task.stop()
        if self._ci_configured:
            self.ci_task.stop()
        self.ai_task.register_done_event(None)

    def _cb_done_event(self, *args):
        # Read data:
        self._read_device()
        return 0

    def wait_until_done(self):
        """Blocks thread until ni_task is finished"""
        self.ai_task.wait_until_done()

        if self._ci_configured:
            self.ci_task.wait_until_done()


class DAQClassFactory(IClassFactory):

    @staticmethod
    def get(name):
        return eval(name)


class TaskLoader(object):

    @staticmethod
    def load(filename: str):
        root_elem = ET.parse(filename).getroot()
        root_type = DAQClassFactory().get(root_elem.get('Type'))
        return root_type.from_xml(root_elem, DAQClassFactory())

    @staticmethod
    def save(item: MadernTask, filename: str):
        if filename != '':
            ET.ElementTree(item.to_xml()).write(filename)


async def main():
    devices = list_devices(verbose=True)

    def _data_callback(change):
        logger.info('Received {} samples'.format(len(change.new)))
    chassis_name = 'MadernRD'

    conf = TaskConfiguration(sampling_rate=100, buffer_size=200,chassis_name=chassis_name,
                             task_name='TestTask2')

    ci_list = [AngularX4Encoder(device_name=f'{chassis_name}Mod5', counter_index=1, name='rotation', pulses_per_rev=2048
                                 )]

    sensors = SensorList(ai_list=[VoltageSensor(device_name='{0}Mod3'.format(chassis_name), channel_index=0, name='test_voltage')
                                  ],
                         ci_list=ci_list)

    task = ContinuousMadernTask(configuration=conf, sensors=sensors)
    task.observe(_data_callback, 'output_data')
    with task:
        task.start()
        for i in range(10):
            await asyncio.sleep(1)
            logger.info('Waiting...')

    for task in asyncio.all_tasks():
        task.cancel()
    await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, filename='example_log.log')
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(stream)

    asyncio.run(main())
