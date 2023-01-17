import madernpytools.data_processing.DAQ as mdaq
import madernpytools.log as mlog
import madernpytools.backbone as mbb
import os, traitlets, logging, sys, asyncio
import numpy as np

logger = logging.getLogger(f'madernpytools.{__name__}')

class ClassFactory(mbb.IClassFactory):
    """ Factory class which allows to generate class instances from this module

    """

    @staticmethod
    def get(name):
        if name in sys.modules[__name__].__dict__:
            return eval(name)
        else:
            for mod in [mdaq]:
                if name in getattr(mod, '__dict__'):
                    return getattr(mod, 'ClassFactory').get(name)
            raise TypeError('Could not find {}'.format(name))


class NIDataLogger(mbb.TraitsXMLSerializer, traitlets.TraitType):
    log = mlog.AbstractLog()
    task = mdaq.ContinuousMadernTask()
    log_size = traitlets.CInt()
    connected = traitlets.CBool(default_value=False)

    def __init__(self, task: mdaq.ContinuousMadernTask=None, log_size: int=None):

        self._log_link = None

        if isinstance(task, mdaq.AbstractMadernTask):
            if log_size is None:
                log_size = int(100 * task.sampling_rate)

            super().__init__(task=task,
                             log_size=log_size,
                             log=mlog.AutoSaveLog(filename=os.path.expanduser('~'),
                                        log_info=mlog.LogInfo(description='',
                                                              sampling_rate=task.sampling_rate,
                                                              signal_header=task.added_keys),
                                                   log_size=log_size),
                             var_names_mapping=[('task', 'task'), ('log_size', 'log_size')])
        else:
            super().__init__()
        self._keep_connected = False

    @traitlets.default('log')
    def _default_log(self):
        return mlog.AutoSaveLog(filename=os.path.expanduser('~'),
                                log_info=mlog.LogInfo(description='',
                                                     sampling_rate=1,
                                                     signal_header=[]),
                                log_size=int(10)
                               )

    @traitlets.observe('log')
    def _log_change(self, change):
        if isinstance(self._log_link, traitlets.link):
            self._log_link.unlink()
        if isinstance(change.new, mlog.AbstractLog):
            self._log_link = traitlets.link((self.task, 'output_data'), (change.new, 'input'))

    @traitlets.default('task')
    def _default_task(self):
        return mdaq.ContinuousMadernTask()

    def connect(self):
        asyncio.create_task(self._start_task())

    def disconnect(self):
        self._keep_connected = False

    async def _start_task(self):

        try:
            with self.task:
                self.task.start()
                logger.info('Started NI-task')
                self.connected = True
                self._keep_connected = True
                while self._keep_connected:
                    await asyncio.sleep(0.1)
                self.task.stop()
        except Exception as err:
            logger.error(err)
        finally:
            self.connected = False
            logger.info('Stopped  NI-task')


def get_default_nilogger():
    chassis_name = 'SimMadernRD'
    gap_sensors = [mdaq.MEcapaNCDTSensor(device_name='{0}Mod3'.format(chassis_name),
                                                           channel_index=i, name=name,
                                         scale=100)
                                       for name, i in [('os_upper', 0), ('os_lower', 1),
                                                       ('ds_upper', 2), ('ds_lower', 3)]
                   ]
    acc_sensors = [
                   # mdaq.PCBAccelerationSensor(device_name='{0}Mod1'.format(chassis_name),
                   #                                               channel_index=i, name=name)
                   #                    for name, i in [('os_acc', 0), ('ds_acc', 1)]
                   ]

    angle_sensor = [mdaq.AngularX4Encoder(device_name=f'{chassis_name}Mod5', counter_index=0,
                                          name='ToolAngle', pulses_per_rev=2048)]

    temp_sensor = [mdaq.Thermocouple(device_name=f'{chassis_name}Mod2',channel_index=1,name='Test',
                                     thermocouple_type=mdaq.nic.ThermocoupleType.J)
                   ]

    sensors = mdaq.SensorList(ai_list=[*gap_sensors, *acc_sensors, *temp_sensor],
                              ci_list=angle_sensor
                              )
    configuration = mdaq.TaskConfiguration(sampling_rate=1024, buffer_size=10240,
                                           chassis_name=chassis_name, task_name='GapTask',
                                           clock_source=f'/{chassis_name}/PFI0'
                                           )

    task = mdaq.ContinuousMadernTask(configuration=configuration, sensors=sensors)
    return NIDataLogger(task, log_size=task.sampling_rate*10)


async def main():

    def _data_callback(change):
        logger.info('Log size {} samples'.format(len(change.new)))

    default_path ='./settings/default_logger.xml'
    if os.path.exists(default_path):
        ni_logger = NIDataLogger.from_xml(mbb.ET.parse(default_path).getroot(),
                                          class_factory=ClassFactory)
    else:
        # Get default task:
        ni_logger = get_default_nilogger()

        # Write default to xml:
        if not os.path.exists(os.path.dirname(default_path)):
            os.mkdir(os.path.dirname(default_path))
        ni_xml = ni_logger.to_xml()
        mbb.ET.ElementTree(ni_xml).write(default_path)

    ni_logger.log.observe(_data_callback, 'data')

    ni_logger.connect()
    logger.info('Connected')
    ni_logger.log.autosave = True
    await asyncio.sleep(2)
    ni_logger.log.autosave = False
    ni_logger.disconnect()
    logger.info('Disconnected')

    for task in asyncio.all_tasks():
        task.cancel()
    await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename='example_log.log')
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s %(threadName)-8s"))
    logger.addHandler(stream)
    mdaq.logger.addHandler(stream)
    mlog.logger.addHandler(stream)

    asyncio.run(main())



