from django.db import models
from daqhats import mcc128, OptionFlags, HatIDs, HatError, AnalogInputMode, AnalogInputRange
from daqhats_utils import select_hat_device, chan_list_to_mask        

class DAQHat(models.Model):
    input_mode_choices = (
        ('SE', 'Single-ended'),
        ('DIFF', 'Differential'),
    )
    options_choices = (
        ('DEFAULT', 'Default'),
        ('CONTINUOUS' , 'Continuous'),
    )
    daq_type_choices = (
        ('MCC_118', 'MCC 118'),
        ('MCC_128', 'MCC 128'),
        ('MCC_134', 'MCC 134'),
        ('MCC_152', 'MCC 152'),
        ('MCC_172', 'MCC 172'),
    )
    input_range_choices = (
        ('BIP_10V', '10V input range'),
        ('BIP_5V', '5V input range'),
        ('BIP_2V', '2V input range'),
        ('BIP_1V', '1V input range'),
    )
    
    name = models.CharField(max_length = 50, default='Default')
    channels = models.CharField(max_length = 50, default='0,1,2,3')
    scan_rate = models.FloatField(default=0)
    options = models.CharField(max_length = 100, choices = options_choices, default='Default')
    samples_per_channel = models.IntegerField(default=0)    
    daq_type = models.CharField(max_length = 100, choices = daq_type_choices, default='MCC 118')
    read_request_size = models.IntegerField(default=0)
    timeout = models.IntegerField(default=0)
    input_mode = models.CharField(max_length = 100, choices=input_mode_choices, default='Single-ended')
    input_range = models.CharField(max_length = 100, choices=input_range_choices, default='10V input range')
    
    def get_channels(self) -> list:
        channel_list = self.channels.split(',')
        for item in range(len(channel_list)):
            channel_list[item] = int(channel_list[item])
        return list(channel_list)
    
    def create_daq(self) -> None:
        input_mode = getattr(AnalogInputMode, self.input_mode)
        input_range = getattr(AnalogInputRange, self.input_range)
        try:
            address = select_hat_device(getattr(HatIDs, self.daq_type))
            self.hat = mcc128(address)
            self.hat.a_in_mode_write(input_mode)
            self.hat.a_in_range_write(input_range)
            print('^^^ Setup MCC 128 HAT device at address', address, '^^^')
        except (HatError, ValueError) as err:
            print('\n', err)

    def start_sensor_scan(self) -> None:
        # Start scan process
        try:
            self.hat.a_in_scan_start(   
                chan_list_to_mask(self.get_channels()), 
                self.samples_per_channel, 
                self.scan_rate,
                getattr(OptionFlags, self.options)
            )
            print('=== Start scanning ===')
        except (HatError, ValueError) as err:
            print('\n', err)
            
    def stop_sensor_scan(self) -> None:
        # Stops and clears running process
        self.hat.a_in_scan_stop()
        self.hat.a_in_scan_cleanup()
        print('=== Scanning stopped ===')
        
    def get_sensor_values(self):
        self.read_result = self.hat.a_in_scan_read(self.read_request_size, self.timeout)

        # Check for an overrun error
        if self.read_result.hardware_overrun:
            print('\n\nHardware overrun\n')
            return 0
        elif self.read_result.buffer_overrun:
            print('\n\nBuffer overrun\n')
            return 0
            
        return self.read_result.data
        
    def __str__(self):
        return self.name
