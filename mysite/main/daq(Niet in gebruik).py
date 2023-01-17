# from daqhats import mcc128, OptionFlags, HatIDs, HatError, AnalogInputMode, \
#     AnalogInputRange
# from daqhats_utils import select_hat_device, chan_list_to_mask

# from main.models import DAQHat

# READ_ALL_AVAILABLE = -1

# class DAQHat():

#     def __init__(self, channels, scan_rate):
#         pass
#         # Options for scan start
#         # self.optionsTest = DAQHat.objects.get("tabletoprdc").options
#         self.channels = channels
#         self.channel_mask = chan_list_to_mask(channels)
#         self.scan_rate = scan_rate
#         self.options = OptionFlags.CONTINUOUS
#         self.samples_per_channel = 0
        
#         # print("type: ", type(HatIDs.MCC_128))
#         # print("type2:", type(0x0146))

#         # Options for scan read
#         self.read_request_size = READ_ALL_AVAILABLE
#         self.timeout = 0

#         # Setting up hat
#         self.input_mode = AnalogInputMode.SE
#         self.input_range = AnalogInputRange.BIP_10V
#         try:
#             address = select_hat_device(getattr(HatIDs,'MCC_128'))
#             self.hat = mcc128(address)
#             self.hat.a_in_mode_write(self.input_mode)
#             self.hat.a_in_range_write(self.input_range)
#             print('^^^ Setup MCC 128 HAT device at address', address, '^^^')
#         except (HatError, ValueError) as err:
#             print('\n', err)

#     def start_sensor_scan(self) -> None:
#         # Start scan process
#         try:
#             self.hat.a_in_scan_start(   
#                 self.channel_mask, 
#                 self.samples_per_channel, 
#                 self.scan_rate,
#                 self.options
#             )
#             print('=== Start scanning ===')
#         except (HatError, ValueError) as err:
#             print('\n', err)

#     def stop_sensor_scan(self) -> None:
#         # Stops and clears running process
#         self.hat.a_in_scan_stop()
#         self.hat.a_in_scan_cleanup()
#         print('=== Scanning stopped ===')

#     def get_hat_values(self):
#         self.read_result = self.hat.a_in_scan_read(self.read_request_size, self.timeout)

#         # Check for an overrun error
#         if self.read_result.hardware_overrun:
#             print('\n\nHardware overrun\n')
#             return 0
#         elif self.read_result.buffer_overrun:
#             print('\n\nBuffer overrun\n')
#             return 0
            
#         return self.read_result.data

#     def get_channels(self) -> list:
#         return self.channels
