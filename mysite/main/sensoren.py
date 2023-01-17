from daqhats import mcc128, OptionFlags, HatIDs, HatError, AnalogInputMode, \
    AnalogInputRange
from daqhats_utils import select_hat_device, chan_list_to_mask
from time import time

READ_ALL_AVAILABLE = -1

class DAQHat():

    def __init__(self, channels, scan_rate):
        self.channels = channels
        self.channel_mask = chan_list_to_mask(channels)
        self.num_channels = len(channels)
        self.index = 0
        
        self.total_samples_read = 0
        self.read_request_size = READ_ALL_AVAILABLE
        self.timeout = 0

        self.options = OptionFlags.CONTINUOUS

        self.resultList = []

        self.start_time = time()
        self.start_scanning_time = time()
        self.buffer = 0

        try:
            address = select_hat_device(HatIDs.MCC_128)
            self.hat = mcc128(address)
            print('^^^ Setup MCC 128 HAT device at address', address, '^^^')
            self.input_mode = AnalogInputMode.SE
            self.input_range = AnalogInputRange.BIP_10V
            self.hat.a_in_mode_write(self.input_mode)
            self.hat.a_in_range_write(self.input_range)
            self.scan_rate = scan_rate
        except (HatError, ValueError) as err:
            print('\n', err)

    def get_channels(self):
        return self.channels

    def start_sensor_scan(self):
        # Setup scan variables
        samples_per_channel = 0
        # Start scan process
        try:
            self.hat.a_in_scan_start(self.channel_mask, samples_per_channel, self.scan_rate,
                                self.options)
            print('=== Start scanning ===')
        except (HatError, ValueError) as err:
            print('\n', err)

    def stop_sensor_scan(self):
        print('=== Scanning stopped ===')
        # Stops and clears running process
        self.hat.a_in_scan_stop()
        self.hat.a_in_scan_cleanup()

    def get_hat_values(self):
        self.start_scanning_time = time()
        self.read_result = self.hat.a_in_scan_read(self.read_request_size, self.timeout)

        # Check for an overrun error
        if self.read_result.hardware_overrun:
            print('\n\nHardware overrun\n')
            return 0
        elif self.read_result.buffer_overrun:
            print(time()-self.start_time)
            print('\n\nBuffer overrun\n')
            return 0
            
        samples_read_per_channel = int(len(self.read_result.data) / self.num_channels)
        self.all_samples = int(len(self.read_result.data))
        self.total_samples_read += samples_read_per_channel
        # print(time() - self.start_scanning_time)

        if samples_read_per_channel > 0:
            self.index = samples_read_per_channel * self.num_channels - self.num_channels
            self.temp_result = self.read_result.data
        else:
            return []
        return self.read_result.data

    def get_number_of_values(self):
        return int(len(self.read_result.data) / self.num_channels)
        # return self.hat.a_in_scan_actual_rate(self.num_channels, self.scan_rate)

    def get_buffer_size(self):
        return self.hat.a_in_scan_status().samples_available
