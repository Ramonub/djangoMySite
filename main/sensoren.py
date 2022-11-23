from daqhats import mcc128, OptionFlags, HatIDs, HatError, AnalogInputMode, \
    AnalogInputRange
from daqhats_utils import select_hat_device, chan_list_to_mask

READ_ALL_AVAILABLE = -1

CURSOR_BACK_2 = '\x1b[2D'
ERASE_TO_END_OF_LINE = '\x1b[0K'

class sensoren:
    def __init__(self, channels):
        self.channels = channels
        self.channel_mask = chan_list_to_mask(channels)
        self.num_channels = len(channels)
        
        self.total_samples_read = 0
        self.read_request_size = READ_ALL_AVAILABLE
        self.timeout = 5.0

        try:
            address = select_hat_device(HatIDs.MCC_128)
            self.hat = mcc128(address)
            print('^^^ Setup MCC 128 HAT device at address', address, '^^^')
            self.input_mode = AnalogInputMode.SE
            self.input_range = AnalogInputRange.BIP_10V
            self.hat.a_in_mode_write(self.input_mode)
            self.hat.a_in_range_write(self.input_range)
        except (HatError, ValueError) as err:
            print('\n', err)

    def startScan(self):
        # Clean other running process if needed
        self.hat.a_in_scan_stop()
        self.hat.a_in_scan_cleanup()
        # Setup scan variables
        samples_per_channel = 0
        options = OptionFlags.CONTINUOUS
        scan_rate = 1000.0
        # Start scan process
        try:
            self.hat.a_in_scan_start(self.channel_mask, samples_per_channel, scan_rate,
                                options)
            print('=== Start scanning ===')
        except (HatError, ValueError) as err:
            print('\n', err)

    def stopScanning(self):
        print('=== Scanning stopped ===')
        # Stops and clears running process
        self.hat.a_in_scan_stop()
        self.hat.a_in_scan_cleanup()

    def getValue(self):
        read_result = self.hat.a_in_scan_read(self.read_request_size, self.timeout)
        if read_result.hardware_overrun:
            print('\n\nHardware overrun\n')
            return 0
        elif read_result.buffer_overrun:
            print('\n\nBuffer overrun\n')
            return 0
            
        samples_read_per_channel = int(len(read_result.data) / self.num_channels)
        self.total_samples_read += samples_read_per_channel

        resultList = []
        if samples_read_per_channel > 0:
            index = samples_read_per_channel * self.num_channels - self.num_channels
            resultList.append(read_result.data[index])
            resultList.append(read_result.data[index+1])
        return resultList
