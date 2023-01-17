import numpy as np
from datetime import datetime
import re, os, csv, time, getpass, asyncio, sys
import traitlets, logging, multiprocessing

import madernpytools.backbone as mbb
from madernpytools.backbone import BackgroundWorker, RepetitiveJob, DataPublisher, DataSubscriber, JobState, \
    SimpleEventSubscriber, ArgumentVerifier, XMLSerializer
import madernpytools.constants as mcnst
from madernpytools.signal_handling import SignalKeyList
import madernpytools.tools.utilities as mutils
from madernpytools.signal_handling import Buffer


logger = logging.getLogger(f'madernpytools.{__name__}')
#logger = multiprocessing.get_logger()


class LogClassFactory(mbb.IClassFactory):

    @staticmethod
    def get(name):
        return eval(name)


class AbstractLogReader(mbb.ProcessStatus):
    """Abstract log reader"""

    def read(self, filename):
        """Reads log_info and data from file"""
        raise NotImplementedError()


class AbstractLogWriter(mbb.ProcessStatus):
    """Abstract log writer"""

    def write(self, filename, log_info, data):
        """Writes log_info and data into file"""
        raise NotImplementedError()


class AbstractLog(traitlets.HasTraits, traitlets.TraitType):
    input = traitlets.TraitType()
    data = traitlets.TraitType()
    n_samples = traitlets.CInt(default_value=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @traitlets.default('input')
    def _input_default(self):
        return mutils.ListofDict([])

    @property
    def info(self):
        """Log information"""
        raise NotImplementedError()

    def add_sample(self, sample):
        raise NotImplementedError()

    def open(self, filename):
        """Open log
        :param filename, filename of the log to open
        """
        raise NotImplementedError()

    def save(self, filename):
        """Open log
        :param filename: filename of the log to open
        :type filename: string
        """
        raise NotImplementedError()

    def reset(self):
        """Reset the log"""
        raise NotImplementedError()


class LogInfo(XMLSerializer):

    def __init__(self, description, sampling_rate, signal_header):
        """Wrapper to hold log information
        :param description: brief description of the log
        :type description: str
        :param sampling_rate: sampling rate used to create the log
        :type sampling_rate: int
        :param signal_header: List with signal descriptions, e.g.: ['Acc1 [m/s^2]', 'Impuls [N]']
        :type signal_header: list
        """

        self._description = description
        self._sampling_rate = sampling_rate
        self._signal_header = signal_header
        XMLSerializer.__init__(self)

        self._info = {'date': ('date (dd/mm/yy)', time.strftime("%d/%m/%Y")),
                      'time': ('time (HH:MM:SS)', time.strftime("%H:%M:%S")),
                      'author': ('author', getpass.getuser()),
                      'description': ('description', description),
                      'sampling_rate': ('sampling_rate [Hz]', sampling_rate),
                      }

    def get_as_list(self):
        """Get log information in list of strings: [[key1, <info of key1>], [key2, <info of key2>]]
        :returns list"""
        return [list(item) for key, item in self._info.items()]

    @property
    def signal_header(self):
        """The signal header of the log info"""
        return self._signal_header

    @signal_header.setter
    def signal_header(self, value):
        """The signal header of the log info"""
        self._signal_header = value

    @property
    def sampling_rate(self):
        return int(float(self['sampling_rate']))

    def keys(self):
        """Returns keys available in the log"""
        return self._info.keys()

    def __getitem__(self, key):
        return self._info[key][1]

    def __setitem__(self, key, value):
        self._info[key] = (key, value)

    def reset(self):
        """Reset date and time of log. """
        self._info['date'] = ('date (dd/mm/yy)', time.strftime("%d/%m/%Y"))
        self._info['time'] = ('time (HH:MM:SS)', time.strftime("%H:%M:%S"))


class CSVLogWriter(AbstractLogWriter):

    def write(self, filename, log_info, data):
        """Writes log_info and data into a log file with the given filename
        :param filename: Filename of the log to write
        :type filename: str
        :param log_info: LogInfo of the log to write
        :type log_info: LogInfo
        :param data: data to write in the log
        :type data: NxD ndarray with N samples and D signals
        """

        self.active = True
        with open(filename, 'w', newline='') as csvfile:
            self.status_message = 'Writing data to {}'.format(filename)
            self.progress=0.0
            logwriter = csv.writer(csvfile, dialect='excel')

            # Log Info
            for item in log_info.get_as_list():
                logwriter.writerow(item)

            # Sample Header:
            logwriter.writerow(log_info.signal_header)

            # Write samples:
            n = len(data)
            for i, sample in enumerate(data):
                logwriter.writerow(sample)
                self.progress = i / n

            self.status_message=''
        self.active = False


class ASCLogReader(AbstractLogReader):

    def read(self, filename):
        """Reads ACS filename and returns its content into log_info and data
        :param filename: Filename of the log to read
        :type filename: str
        :return: log information and data
        :rtype: LogInfo, numpy.ndarray
        """
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            data = []
            header = []
            for row in reader:
                try:
                    row_tmp = row
                    row_tmp[0] = row[0][8:]
                    row_tmp = [s.replace(',', '.') for s in row_tmp[:-1]]
                    row_tmp = np.array(row_tmp, dtype=np.float64)
                    if row_tmp.size != 0:
                        data.append(row_tmp)
                except RuntimeError:
                    header.append(row)

        # Store header:
        log_info = LogInfo(description="", sampling_rate=0, signal_header=[])
        for h in header[:-1]:
            found_key = False
            # Check if in header:
            for key in log_info.keys():
                pattern = ''.join(['{0}.*'.format(item) for item in re.split('_| ', key)])
                if re.match(pattern, h[0], flags=re.I):
                    log_info[key] = h[1]
                    found_key = True
                    break

            if not found_key:
                # Non-default entry:
                log_info[h[0]] = h[1]

        # Store signal names:
        log_info.signal_header = header[-1]

        # log data:
        data = np.vstack(data)

        return log_info, data


class CSVLogReader(AbstractLogReader):

    def read(self, filename):
        """Read csv log
        :param filename: the filename to read
        :return: Log information and logged data
        :rtype: (LogInfo, numpy.ndarray)
        """

        self.active = True
        with open(filename, 'r', newline='') as csvfile:
            self.status_message = 'Reading {}'.format(filename)
            self.progress=0.0
            logreader = csv.reader(csvfile, dialect='excel')

            found_data = False
            header = []
            data = []

            for i, row in enumerate(logreader):
                if found_data:
                    data.append(np.array(row, dtype=np.float64))
                else:
                    try:
                        data.append(np.array(row, dtype=np.float64))
                        found_data = True
                    except ValueError:
                        header.append(row)

                if (i % 100) == 0:
                    self.progress = (i % 5000) / 5000

        # Store header:
        log_info = LogInfo("", 0, [])
        for h in header[:-1]:
            found_key = False
            # Check if in header:
            for key in log_info.keys():
                pattern = ''.join(['{0}.*'.format(item) for item in re.split('_| ', key)])
                if re.match(pattern, h[0], flags=re.I):
                    log_info[key] = h[1]
                    found_key = True
                    break

            if not found_key:
                # Non-default entry:
                log_info[h[0]] = h[1]

        # Store signal names:
        log_info.signal_header = header[-1]

        # log data:
        log_data = np.vstack(data)
        self.status_message = ''
        self.active = False

        return log_info, log_data


class SiemensLog(object):

    def __init__(self, info:LogInfo, signal_dict=None):
        self._info = info
        self._signal_dict = signal_dict

    def get_signal(self, key, include_time=True):
        # Gather values and time:
        time = []
        values = []

        for sample in self._signal_dict[key]:
            values.append(sample['VarValue'])
            time.append(sample['TimeString'])

        sort_ind = np.argsort(time)
        values = np.array(values)[sort_ind]
        time = np.array(time)[sort_ind]

        if include_time:
            return [time, values]
        else:
            return values

    def get_signal_pair(self, x_key, y_key):

        # Get signals
        [x_time, x_vals] = self.get_signal(x_key)
        [y_time, y_vals] = self.get_signal(y_key)

        # Get union of signal:
        time_values = sorted(list(set(x_time).intersection(set(y_time))))
        x_time = sorted(x_time)
        y_time = sorted(y_time)
        x_vals = [x_vals[i] for i, item in enumerate(x_time) if item in time_values]
        y_vals = [y_vals[i] for i, item in enumerate(y_time) if item in time_values]

        return [x_vals, y_vals]

    @staticmethod
    def load(filename):
        return SiemensLog(*SiemensLogReader().read(filename))

    def keys(self):
        return self._signal_dict.keys()

    def __getitem__(self, item):
        return self.get_signal(item)

    @property
    def info(self):
        return self._info


class BoschLog(object):

    def __init__(self, info, data):
        self._info = info
        self._data = data
        self._signal_dict = {}

        # create signal dictionary
        self._signal_dict = {}
        for i, name in enumerate(self._info.signal_header):
            # Filter NaN:o
            time_ind = (np.isnan(self._data[:, i])==False)
            self._signal_dict[name] = (self._data[time_ind, 0], self._data[time_ind, i])

    def get_signal(self, key, include_time=True):
        if include_time:
            return self._signal_dict[key]
        else:
            return self._signal_dict[key][1]

    def get_signal_pair(self, x_key, y_key):

        x_data = self.get_signal(x_key)
        y_data = self.get_signal(y_key)

        # Find samples that were generated at the same time
        x_time = x_data[0] # n-dimensional
        y_time = y_data[0] # m-dimensional

        # Set-based selection of data:

        # Get intersection of time stamps such that we can correlate data points of x and y data:
        time_values = sorted(list(set(x_time).intersection(set(y_time))))
        x_vals = np.array([x_data[1][i] for i, item in enumerate(x_data[0]) if (item in time_values)])
        y_vals = np.array([y_data[1][i] for i, item in enumerate(y_data[0]) if (item in time_values)])

        return [x_vals, y_vals]

    @staticmethod
    def load(filename=''):
        info, data = BoschLogReader().read(filename)
        return BoschLog(info, data)

    def __getitem__(self, item):
        return self.get_signal(item)

    @property
    def data(self)->np.ndarray:
        return self._data

    @property
    def info(self) -> LogInfo:
        return self._info


class BoschLogReader(AbstractLogReader):

    def read(self, filename):
        """Read bosch log
        :param filename: the filename to read
        :return: Log information and logged data
        :rtype: (LogInfo, numpy.ndarray)
        """
        with open(filename, 'r', newline='') as textfile:
            logreader = csv.reader(textfile, delimiter=' ')

            found_data = False
            header = []
            data = []
            for row in logreader:
                if found_data:
                    row_tmp = ['nan' if s == '' else s for s in row]
                    data.append(np.array(row_tmp, dtype=np.float64))
                else:
                    try:
                        row_tmp = ['nan' if s == '' else s for s in row]
                        data.append(np.array(row_tmp, dtype=np.float64))
                        found_data = True
                    except ValueError:
                        header.append(row)

        # Store header:
        log_info = LogInfo("", 0, [])
        for h in header[:-1]:
            found_key = False
            # Check if in header:
            for key in log_info.keys():
                pattern = ''.join(['{0}.*'.format(item) for item in re.split('_| ', key)])
                if re.match(pattern, h[0], flags=re.I):
                    log_info[key] = h[1]
                    found_key = True
                    break

            if not found_key:
                # Non-default entry:
                log_info[h[0]] = h[1]

        # Store signal names:
        log_info.signal_header = header[-1][:-1]

        # log data:
        log_data = np.vstack(data)

        return BoschLog(log_info, log_data)


class SiemensLogReader(AbstractLogReader):

    def __init__(self):
        self._conversion_dict = {'VarName': lambda obj: str(obj),
                                 'TimeString': lambda obj: datetime.strptime(obj, '%d/%m/%Y %I:%M:%S %p').timestamp(),
                                 'VarValue': lambda obj: float(obj) / 10,
                                 'Validity': lambda obj: bool(obj),
                                 'Time_ms': lambda obj: float(obj)}

    def read(self, filename):
        """Siemens log file
        :param filename: the filename to read
        :return: Log information and logged data
        :rtype: (LogInfo, signal_dict)
        """

        with open(filename, 'r', newline='') as textfile:
            logreader = csv.reader(textfile, delimiter=',')

            signals = {}

            i = 0
            for row in logreader:
                if i == 0:
                    self._header = row
                else:
                    sample = self.convert_line(row)
                    if sample['VarName'] in signals:
                        signals[sample['VarName']].append(sample)
                    else:
                        signals[sample['VarName']] = [sample]
                i += 1

        # Collect time samples:
        time_set = set([])
        for var_name, item in signals.items():
            time_list = []
            for sample in item:
                time_list.append(sample['TimeString'])
            time_set = time_set.union(set(time_list))

        signals['time'] = []
        for t in list(time_set):
            signals['time'].append({'TimeString': t, 'VarValue': t, 'Validity': True, 'Time_ms': 0})
        self._header.insert(0, 'time')

        log_info = LogInfo('Siemens log file', sampling_rate=-1, signal_header=list(signals.keys()))

        return SiemensLog(log_info, signals)

    def convert_line(self, line):
        row_dict = dict(zip(self._header, line))

        for key, item in row_dict.items():
            row_dict[key] = self._conversion_dict[key](item)

        return row_dict


class Log(mbb.ProcessStatus, AbstractLog):

    def __init__(self,  log_info):
        """Creates a Log object
        :param log_info: Log information.
        :type log_info: LogInfo
        """
        # Variables required for initialization
        self._log_info = log_info
        super().__init__(data=mutils.ListofDict([]))

    def save(self, filename):
        """ Save log

        :param filename: task_name of log file.
        :type filename: str
        """

        # Define writer
        logger.info('Saving to {}'.format(filename))
        writer = CSVLogWriter()
        writer.link_progress_with(self)

        # Get data:
        if isinstance(self.data, mutils.Listof):
            data = np.vstack([self.data.get_key_values(key) for key in self.data.get_keys()]).T.astype(dtype=np.float32)
        elif isinstance(self.data, (np.ndarray, list)):
            data = self.data
        else:
            raise RuntimeError('Unknown data format')
        writer.write(filename, self._log_info, data)
        writer.unlink_progress_with(self)

    def open(self, filename):
        """ Open log

        :param filename: filename of log to open.
        """
        # Check extensions:
        if re.match('.*\.csv', filename):
            reader = CSVLogReader()
        elif re.match('.*\.acs', filename):
            reader = ASCLogReader()
        else:
            raise RuntimeError('Unknown file extension, can only read CSV or ASC files')

        # Link progress:
        reader.link_progress_with(self)
        self._log_info, self._data = reader.read(filename)
        reader.unlink_progress_with(self)

    @property
    def info(self):
        """ Log info of this log

        :return: Log info
        :rtype: LogInfo
        """
        return self._log_info

    @traitlets.observe('input')
    def _input_change(self, change):
        if self.active:
            self.add_sample(change.new)

    def add_sample(self, sample):
        """ Add sample to log
        :param sample: sample(s) to add to log
        :type sample: D or NxD numpy array, with N the number of samples and D the number of signals to log
        """
        if isinstance(sample, dict):
            self.data.append(sample)
        elif isinstance(sample, (list, mutils.ListofDict)):
            self.data += sample
        else:
            raise RuntimeWarning('Received unknown data type, cannot add sample to log')

        self.n_samples = len(self.data)

    def reset(self, reset_time: bool = True, clear_data: bool = True) -> None:
        """Clear log data and reset time of log information
        :param reset_time: If true, sets data and time to now (default=True)
        :param clear_data: If true, clears the log data (default=True)
        """
        if clear_data:
            self.data = mutils.ListofDict()
            self.n_samples = 0
        if reset_time:
            self.info.reset()


def write_file(q):

    while True:
        logger.info(f'Checking que')
        next_item = q.get()
        if isinstance(next_item, tuple):
            filename, file_header, signal_header, data = next_item
            logger.info(f'Writing to {filename}')
            with open(filename, 'w', newline='') as csvfile:
                logwriter = csv.writer(csvfile, dialect='excel')

                # Log Info
                for item in file_header:
                    logwriter.writerow(item)

                # Sample Header:
                logwriter.writerow(signal_header)

                # Write samples:
                for sample in data:
                    logwriter.writerow(sample)

        elif next_item == 'DONE':
            break


class AutoSaveLog(XMLSerializer, AbstractLog, mbb.ProcessStatus):
    input = mutils.ListofDict()
    buffer = mutils.ListofDict()
    save_counter = traitlets.CInt(default_value=0)
    unsaved_samples = traitlets.CInt(default_value=0)

    def __init__(self, filename: str, log_info: LogInfo, log_size: int = 10000, buffer_size: int = None) -> None:
        """Initialize Autosave log

        :param filename: Filename to use for auto saved logs, e.g. example_log.csv will result in log files
                         example_log000000.csv, example_log000001.csv etc...
        :param log_info: Information about the log
        :param log_size: A log file is saved every time this number of samples is reached.
        :param buffer_size: Size of the log buffer, this value should be greater than log_size.
                            If not provided, it defaults to 2*log_size
        """

        # Define variables for xml serialization:
        self._filename = ''
        self._log_info = log_info
        self._log_size = ArgumentVerifier(int, 10000).verify(log_size)
        self._buffer_size = ArgumentVerifier(int, log_size*2).verify(buffer_size)

        # Initialize inhereted classes
        super().__init__()

        # Class initialization
        (root, _) = os.path.splitext(filename)
        self.set_filename(filename)

        # Buffer for the log:
        self._buffer = Buffer(n=self._buffer_size,
                              signal_keys=SignalKeyList(log_info.signal_header))
        self._sample_count = 0
        self.active = False

        self._queue = multiprocessing.Queue()
        self._proc = None

        traitlets.link((self._buffer, 'output_data'), (self, 'buffer'))

    @traitlets.observe('active')
    def _active_change(self, change):
        logger.info('Status Active: {}'.format(change.new))

    @property
    def autosave(self):
        return self.active

    @autosave.setter
    def autosave(self, value):
        logger.info('Autosave change')
        if value:
            logger.info('Starting Autosave')
            self._proc = multiprocessing.Process(target=write_file, args=(self._queue,))
            self._proc.start()

            # Reset unsaved samples to 0
            self.unsaved_samples = 0
        elif (not value) and self.active:
            logger.info('Stopping auto save')
            # Safe remainder
            if self.unsaved_samples > 0:
                logger.info('Saving remainder...')
                data = self._buffer.get_n_latest(self.unsaved_samples)
                logger.info('Saving remainder...')
                asyncio.create_task(self._write_samples_process(data, self._getfilename(), close_queue=True))


        # Set autosave flag:
        self.active = value

    def ready(self, overwrite_files: bool = False) -> mcnst.Errors:
        """
        :param overwrite_files: Allow existing files to be overwritten
        :return: if autosavelog is ready to run.
        """

        # Get filename template
        fn_template = '{0}\d{{6,6}}'.format(os.path.basename(self._filename))  # Example: test000001.csv

        # Get file dir
        file_dir = os.path.dirname(self._filename)
        if file_dir == '':
            file_dir = './'

        # List files that match template:
        files = [f for f in os.listdir(file_dir) if re.match(fn_template, f)]
        ready_to_start = mcnst.Errors.no_error

        # Check for clash:
        if overwrite_files and (len(files) > 0):
            for f in files:
                os.remove('{0}\\{1}'.format(file_dir, f))
        elif len(files) > 0:
            ready_to_start = mcnst.Errors.file_exists

        return ready_to_start

    @traitlets.observe('input')
    def _input_change(self, change):
        logger.info('Got {} samples'.format(len(change.new)))
        self.add_sample(change.new)

    def add_sample(self, samples: mutils.ListofDict):
        """ Add sample to log
        :param samples: Sample(s) to add to log
        :type samples: D or NxD numpy array, with N the number of samples and D the number of signals to log
        """
        # Get sample size:
        if isinstance(samples, dict):
            samples = mutils.ListofDict([samples])
        n_samples = len(samples)

        # Add new samples to buffer:
        self._buffer.add_data(samples)

        # Update sample counter:
        self.n_samples += n_samples

        # Keep track of unsaved samples:
        if self.active:
            self.unsaved_samples += n_samples
            logger.info('Updated unsaved samples to {}'.format(self.unsaved_samples))

    @traitlets.observe('unsaved_samples')
    def _unsaved_sample_change(self, change):
        if self.unsaved_samples >= self._log_size:
            # Get samples
            data = self._buffer.get_n_latest(self._log_size)

            # Write
            #asyncio.create_task(self._write_samples_async(data, self._getfilename()))
            asyncio.create_task(self._write_samples_process(data, self._getfilename()))
            #self._write_samples(data, self._getfilename())
            self.save_counter += 1  # Save file counter

            # Narrow number of unsaved samples:
            self.unsaved_samples -= len(data) # Reset sample counter

    def _getfilename(self):
        # Create filename
        return '{0}{1:0>6}.csv'.format(
            self._filename,
            self.save_counter)

    async def _write_samples_async(self, data, filename):

        self._log_info.reset()      # Update time
        log = Log(self._log_info)
        log.data = data
        await asyncio.to_thread(log.save(filename))

    async def _write_samples_process(self, data, filename, close_queue=False):
        """

        @param data:  Data to write
        @param filename: Filename to write to
        @param close_queue: Close Queue when done
        @return:
        """
        self._log_info.reset()      # Update time

        def convert_data(data):
            return np.vstack([data.get_key_values(key) for key in data.get_keys()]).T.astype(dtype=np.float32)

        data_array = await asyncio.to_thread(convert_data, data)
        info_list = self._log_info.get_as_list()
        sig_header = self._log_info.signal_header
        logger.info(f'Putting {filename} to log writer (size: {data_array.shape})')
        self._queue.put((filename, info_list, sig_header, data_array))
        if close_queue:
            self._queue.put('DONE')

    def reset(self, reset_time: bool = True, clear_data: bool = True) -> None:
        """Clear log data and reset time of log information
        :param reset_time: If true, sets data and time to now (default=True)
        :param clear_data: If true, clears the log data (default=True)
        """
        if clear_data:
            self._buffer.reset()
            self.save_counter = 0       # Count save files
            self.unsaved_samples = 0  # Count unsaved samples
        if reset_time:
            self.info.reset()

    @property
    def filename(self):
        """ Returns filename

        """
        return self._filename

    def __del__(self):
        if self._proc :
            if self._proc.is_alive():
                self._proc.terminate()

    @filename.setter
    def filename(self, value):
        """ Set filename

        """
        self.set_filename(value)

    def set_filename(self, filename):
        """Set filename"""
        (root, _) = os.path.splitext(filename)
        self._filename = root

    @property
    def n_signals(self):
        """Total number of signals in log"""
        return len(self._buffer.output.get_keys())

    @property
    def buffer_size(self):
        """Size of the log buffer

        :return: size of the log buffer
        """
        return self._buffer.buffer_size

    @property
    def info(self):
        """

        :return: Log information
        """
        return self._log_info

    @property
    def data(self):
        """
        :return: Log buffer
        """
        return self._buffer.data

    @data.setter
    def data(self, value):
        """Sets the buffer data

        :param value: buffer data, dimensions should match buffer size
        :return:
        """
        if self._buffer.shape == value.shape:
            self._buffer = value
        else:
            raise RuntimeError("Provided data does not match buffer size {0}".format(self._buffer.shape))

    def save(self, filename):
        """

        :param filename: Filename of log
        :return: None
        """
        CSVLogWriter().write(filename, self._log_info, self.data)

    def open(self, filename):
        """

        :param filename: Filename of log to open
        :return:
        """
        raise NotImplementedError("Open not available for autosave log.")


class LogInterpreter(object):

    def __init__(self, data_log):
        """

        :param data_log:
        :type data_log: log.AbstractLog
        """
        self._log = data_log

    @property
    def log(self):
        return self._log

    def get_n_latest(self, n):
        raise NotImplementedError()

    def get_new(self):
        raise NotImplementedError()

    def get_all(self):
        raise NotImplementedError()


class LogTracker(LogInterpreter, DataPublisher):

    def __init__(self, data_log):
        # Initialize base classes:
        DataPublisher.__init__(self)
        LogInterpreter.__init__(self, data_log)

        # Track variables:
        self._last_read = 0
        self._last_cb_read = 0

        # Setup job and worker:
        self._job = RepetitiveJob(rate=100)
        self._job.work = self._check_for_new_data
        self._job.setup = self._setup_tracking
        self._worker = BackgroundWorker(self._job)

    @property
    def track_changes(self):
        return self._job.state == JobState.Active

    @track_changes.setter
    def track_changes(self, value):
        if value and not (self._job.state == JobState.Active):
            self._worker.start()
        elif not value and (self._job.state == JobState.Active):
            self._worker.stop()

    @property
    def tracking_rate(self):
        return self._job.rate

    @tracking_rate.setter
    def tracking_rate(self, value):
        self._job.rate = value

    def _setup_tracking(self):
        self._cb_read = self._log.n_samples

    def _check_for_new_data(self):
        new_data = self._get_new(self._last_cb_read)

        if new_data is not None:
            # Add samples
            self.raise_event(new_data)
            self._last_cb_read += new_data.shape[0]

    def get_n_latest(self, n):
        """Returns the last n samples"""
        if self._log.n_samples > 0:
            data = self._log.data
            self._last_read = self._log.n_samples
            return data[-min(n, data.shape[0]):, :]
        else:
            return None

    def get_new(self):
        """Get the samples from the log since last manual read"""

        # Check for new data:
        newdata = self._get_new(self._last_read)

        # Update last read value:
        self._last_read = self._log.n_samples if newdata is not None else self._last_read

        return newdata

    def _get_new(self, n_last_read):
        """Get the samples from the log since last read"""

        # Get the new number of samples
        if self._log.n_samples > 0:
            n_prev = n_last_read         # Sample count on last read
            n_now = self._log.n_samples  # Sample count currently
            n_new = min(self._log.data.shape[0], n_now-n_prev)  # Number new samples to read (limited to buffer size)
        else:
            n_new = 0

        if n_new > 0:
            new_data = self._log.data[-n_new:, :]
            # update last read:
        else:
            new_data = None

        return new_data

    def get_all(self):
        return self._log.data


class LogDataVerifier(DataSubscriber, DataPublisher):

    def __init__(self, use_trigger=True, use_limit=True):
        """Class checks if data meets trigger and log limit requirements"""
        DataSubscriber.__init__(self)
        DataPublisher.__init__(self)

        # Trigger and limit:
        self._limit = LogLimit()
        self._trigger = LogTrigger()

        # Create internal events
        self._event_data_received = DataPublisher()                # Internal event that raises on new data received
        self._sub_data_verified = SimpleEventSubscriber(self._cb_data_verified)  # Internal subscriber to receive verified data

        # Setup data flow:
        # New data is directly passed to self._trigger.update(..), then
        if use_limit and use_trigger:
            # Pass new data through trigger -> limit -> subscribers
            self._new_data_cb = self._trigger.cb_new_data
            self._trigger.event_new_data.connect(self._limit)             # Stream data from trigger to limit
            self._limit.event_new_data.connect(self._sub_data_verified)   # Stream data from trigger to verified data subscriber
        elif use_limit:
            # Pass new data through limit -> subscribers
            self._new_data_cb = self._limit.cb_new_data
            self._limit.event_new_data.connect(self._sub_data_verified)
        elif use_trigger:
            # Pass new data through trigger -> subscribers
            self._new_data_cb = self._trigger.cb_new_data
            self._trigger.event_new_data.connect(self._sub_data_verified)
        else:
            # Pass new data to subscribers
            self._new_data_cb = self._cb_data_verified

    @property
    def limit(self):
        """ Data Limiter
        """
        return self._limit

    @property
    def trigger(self):
        """ Data Trigger
        """
        return self._trigger

    def _cb_data_verified(self, publisher):
        """ Raises data event

        :param publisher:
        :return:
        """
        self.raise_event(publisher.get_data())

    def cb_new_data(self, publisher):
        self._new_data_cb(publisher)


class AbstractLimit(traitlets.HasTraits):
    input = mutils.ListofDict()
    output = mutils.ListofDict()

    limit = traitlets.CInt(default_value=0)
    limit_reached = traitlets.Bool(default_value=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        raise NotImplementedError()

    @traitlets.default('input')
    def _input_default(self):
        return mutils.ListofDict([])

    @traitlets.default('output')
    def _input_default(self):
        return mutils.ListofDict([])


class LogLimit(AbstractLimit):

    def __init__(self, limit=0):
        super().__init__(limit=limit)
        self._cnt = 0

    @traitlets.observe('limit')
    def _limit_chang(self, change):
        self.reset()

    @property
    def sample_count(self):
        return self._cnt

    def reset(self):
        """
        Resets the sample counter.
        :return:
        """
        self.limit_reached = False
        self._cnt = 0

    @traitlets.observe('input')
    def _input_change(self, change):
        if not self.limit_reached:
            n_samples = len(change.new)

            if (n_samples + self._cnt) < self.limit:
                # Limit enabled:
                self._cnt += n_samples  # Update counter
                self.output = change.new
            else:
                # Forward remaining data:
                ind = int(min(self.limit - self._cnt, n_samples))
                self._cnt += ind  # Update counter
                self.output = change.new[:ind]
                self.limit_reached = True


class AbstractTrigger(traitlets.HasTraits):
    input = traitlets.TraitType()
    output = traitlets.TraitType()
    triggered = traitlets.Bool(default_value=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def wait_for_trigger(self):
        raise NotImplementedError()

    def set_trigger(self, *args):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    @traitlets.default('input')
    def _input_default(self):
        return mutils.ListofDict([])

    @traitlets.default('output')
    def _input_default(self):
        return mutils.ListofDict([])


class LogTrigger(AbstractTrigger):
    trigger_key = traitlets.CUnicode(default_value='')
    trigger_value = traitlets.CFloat(default_value=0.0)

    def __init__(self, trigger_value=0.0, signal_key=None, trigger_condition=None) -> None:
        """ Trigger functionality for log

        :param trigger_value: Value which triggers event_new_data to raise on new samples (default 0.0)
        :param signal_index:  index of the signal to which the trigger applies  (default -1, no trigger)
        :param trigger_condition: condition of the trigger (default: np.greater_equal)
        """
        self._condition = None
        self.set_trigger(signal_key=signal_key,
                         value=trigger_value,
                         condition=trigger_condition)

        super().__init__(input=mutils.ListofDict(), output=mutils.ListofDict())

    @traitlets.observe('input')
    def _input_change(self, change):

        if self.triggered:
            self.output = self.input
        else:
            if any(self._condition(change.new.get_key_values(self.trigger_key),
                                   self.trigger_value)):
                self.triggered = True
                self.output = self.input

    def set_trigger(self, signal_key, value: float = 0.0, condition: object = None) -> None:
        """

        :param signal_key: The index for which the trigger should be active, if set to -1, trigger is disabled
        :param value: The value on which the trigger should activate
        :param condition: The condition for which the value should trigger (defaults: to signal >= value)
        :return: None
        """
        self.trigger_key = signal_key
        self.trigger_value = value
        self._condition = np.greater_equal if condition is None else condition
        self.reset()

    def reset(self):
        """Reset triggered state (triggered remains true if trigger_key is '')"""

        if self.trigger_key != '':
            # Use trigger functionality
            self.triggered = False
        else:
            # Ignore trigger
            self.triggered = True


class RandomGenerator(traitlets.HasTraits):
    output = mutils.ListofDict()

    def __init__(self, keys: list):
        super().__init__(output=mutils.ListofDict())
        self._cnt = 0
        self._keys = ['count'] + keys

    def reset(self):
        self._cnt = 0
        self.output = mutils.ListofDict()

    def generate(self, n_samples):
        values = np.arange(n_samples) + self._cnt
        random = np.random.randn(n_samples, len(self._keys) - 1 )
        data = np.hstack([values[:, None], random])

        self._cnt += n_samples
        self.output = mutils.ListofDict(items=[dict(zip(self._keys, line)) for line in data])



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename='example_log.log')
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(stream)


    logger.info('Testing Log')
    myinfo = LogInfo(description="testinfo",
                     sampling_rate=10,
                     signal_header=['count', 'A', 'B'])

    logger.info('Saving xml data')
    mutils.XMLFileManager.save(myinfo, 'test_info.xml')
    myinfo.signal_header

    logger.info('loading xml data')
    my_info = mutils.XMLFileManager.load('test_info.xml', LogClassFactory)

    mylog = Log(myinfo)
    for i in range(100):
        mylog.add_sample(dict(zip(my_info.signal_header, [i+1] + list(np.random.randn(2)))))
        time.sleep(0.01)

    # Save and open:
    mylog.save('./data/test.csv')
    mylog.open('./data/test.csv')
    mylog.reset()

    #
    mygen = RandomGenerator(keys=['A', 'B'])
    limit = LogLimit(20)
    trigger = LogTrigger(trigger_value=20, signal_key='count')

    # Create connections:
    links = [
            traitlets.link((mygen, 'output'), (trigger, 'input')),
            traitlets.link((trigger, 'output'), (limit, 'input')),
            traitlets.link((limit, 'output'), (mylog, 'input')),
            ]


    def trigger_call(change):
        logger.info(f'{change.task_name}: {change.new}')

    def samples_received(change):
        logger.info(f'{change.task_name} received {len(change.new)} samples')

    # Observe log
    trigger.observe(trigger_call, 'triggered')
    limit.observe(trigger_call, 'limit_reached')
    mylog.observe(samples_received, 'data')

    for i in range(10):
        mygen.generate(n_samples=10)

    for l in links:
        l.unlink()

    logger.info('----- Testing Autosave log')

    fn = './data/auto_save_test.csv'
    autosave_log = AutoSaveLog(filename=fn,
                               log_info=myinfo,
                               log_size=20
                               )

    def save_change(change):
        logger.info('Save change: {0}'.format(change.new))

    autosave_log.observe(save_change, 'save_counter')

    mygen.reset()
    links = [
        traitlets.link((mygen, 'output'), (autosave_log, 'input')),
    ]
    autosave_log.autosave = True

    async def test():
        for i in range(100):
            mygen.generate(2)
            await asyncio.sleep(0.05)
    asyncio.run(test())
    autosave_log.autosave = False

