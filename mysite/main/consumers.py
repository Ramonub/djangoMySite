import json, pytz
from time import time
from shutil import copyfile
from datetime import datetime
import asyncio, os
# from main.ftsensor import ftSensor

import traitlets
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.generic.websocket import AsyncConsumer
from asgiref.sync import sync_to_async

import madernpytools.log as mlog
from madernpytools.signal_handling import ISignalKeyList
import madernpytools.signal_handling as msigs

from mysite import settings
from main.models import DAQHat


class WSConsumer(AsyncWebsocketConsumer, traitlets.HasTraits):

    async def connect(self) -> None:
        print('!!! Consumer connected !!!')
        self.room_group_name = 'tabletoprdc'
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.channel_layer.send('sensor_stream', {'type': 'activate', 'group_name': self.room_group_name})
        
        self.log_active = False
        self.sensor_values = []
        self.timestep = 0
        self.start_time = 0
        self.number_of_values = 0
        self.scan_rate = 0
        self.number_of_message = 0
        self.time_when_started = time()
        self.list_of_logs = []
        self.list_of_logs_info = []
        await self.accept()
        await self._send_logs()

    async def disconnect(self, close_code) -> None:
        print('!!! Consumer disconnected !!!')
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data) -> None:
        text_data_json = json.loads(text_data)
        if text_data_json['logging'] == 'True':
            # Creating and starting log
            print('!!! Enabled logging !!!')
            log_info = mlog.LogInfo(description=self.room_group_name,
                                    signal_header=self.added_keys,
                                    sampling_rate=self.scan_rate
                                    )
            self.log = mlog.Log(log_info)
            self.log.active = True
            self.log_active = True
            await self.send(text_data=json.dumps({
                'file_info': '',
            }))
        else:
            # Disable and save log
            print('!!! Disabled logging !!!')
            self.log.active = False
            self.log_active = False

            file_name = self.room_group_name + '_log_' + datetime.now(pytz.timezone('Europe/Amsterdam')).strftime('%H%M%S') + '.csv'
            file_location = settings.MEDIA_ROOT + file_name
            print("file", file_location)
            self.log.save(file_name)
            copyfile(file_name, file_location)
            os.remove(file_name)
            await self._send_logs()

    async def recieve_message_group(self, event) -> None:
        self.number_of_sensors = event['number_of_sensors']
        self.sensor_values = event['sensor_values']
        self.scan_rate = event['scan_rate']
        self.timestep = 1/self.scan_rate

        # Logging
        if self.log_active:
            for sensor_value_number in range(0, len(self.sensor_values), self.number_of_sensors):
                # samples = {'Rate': self.number_of_values, 'Timestep': self.start_time + (self.timestep * self.number_of_values)}
                samples = {}
                self.number_of_values += 1
                for signal in range(self.number_of_sensors):
                    samples['signal ' + str(signal)] = self.sensor_values[sensor_value_number + signal]
                self.log.input = samples
        else:
            self.start_time = 0

        # Sensor value's for presentation
        self.sensor_value = []
        for number in range(self.number_of_sensors):
            self.sensor_value.append(self.sensor_values[int(number)])

        # Sending data to Django web interface
        self.number_of_message += 1
        await self.send(text_data=json.dumps({
            'sensor_values': self.sensor_value,
            'message_number': self.number_of_message,
            'running_time': time() - self.time_when_started,
        }))
            
    async def _send_logs(self):
        self.update()
        # await sync_to_async(self.logModel.save)()
        await self.send(text_data=json.dumps({
            'file_info': self.list_of_logs_info,
            'list_of_file_names': self.list_of_logs
        }))
       
    def update(self):
        list_of_files = [f for f in os.listdir(settings.MEDIA_ROOT) if os.path.isfile(os.path.join(settings.MEDIA_ROOT,f))]
        number_of_files = len(list_of_files)
        self.list_of_logs = []
        self.list_of_logs_info = []
        
        max_number_of_files = 15
        if number_of_files < max_number_of_files:
            max_number_of_files = number_of_files
        
        # Put the 15 newest files in a list
        for i in range(max_number_of_files):
            temp_newest_file = 0
            for j in range(len(list_of_files)):
                file_path = (os.stat(str(settings.MEDIA_ROOT + list_of_files[j])).st_ctime)
                if file_path > temp_newest_file:
                    temp_newest_file = file_path
                    highest_temp = list_of_files[j]
                    
            list_of_files.remove(highest_temp)
            self.list_of_logs.append(settings.MEDIA_URL + highest_temp)
            self.list_of_logs_info.append(str(datetime.fromtimestamp(temp_newest_file, pytz.timezone('Europe/Amsterdam')).strftime('Log file from: %d-%m-%Y %H:%M:%S')))
            
        # Remove files that are not the newest 15
        for file in list_of_files:
            os.remove(settings.MEDIA_ROOT + file)

    @property
    def added_keys(self) -> ISignalKeyList:
        # list_of_keys = ['Timestep']
        list_of_keys = []        
        for signal in range(self.number_of_sensors):
            list_of_keys.append('signal ' + str(signal))
        return msigs.SignalKeyList(list_of_keys)



class WSManager(AsyncConsumer):

    def __init__(self, *args, **kwargs) -> None:
        self.active = False
        AsyncConsumer.__init__(self, *args, **kwargs)

    async def activate(self, event) -> None:
        if self.active == False:
            print('<<< Manager activated >>>')
            self.DAQSystem = await sync_to_async(DAQHat.objects.first)()
            self.DAQSystem.create_daq()                     # Set DAQ HAT settings
            self.DAQSystem.start_sensor_scan()              # Starting sensor scan on DAQ HAT
            
            # self.ftSensor = ftSensor()
            # self.ftSensor.getNumberOfConnectedSensors()
            
            self.active = True
            asyncio.create_task(self.run_sensor(self.DAQSystem, event.get('group_name')))   # Create task with both system

    async def deactivate(self, event) -> None:
        print('<<< Manager stopped >>>')
        self.active = False
        self.running = False
        self.DAQSystem.stop_sensor_scan()                   # Stopping sensor scan on DAQ HAT
    
    async def run_sensor(self, system, group_name) -> None:
        self.room_group_name = group_name
        self.running = True

        while self.running:
            self.list_of_values = system.get_sensor_values()
            # print(len(self.list_of_values))

            if self.list_of_values != []:
                self.number_of_sensors = len(system.get_channels())
                await self.channel_layer.group_send(self.room_group_name, {
                    'type': 'recieve_message_group',
                    'scan_rate': self.DAQSystem.scan_rate,
                    'sensor_values': self.list_of_values,
                    'number_of_sensors': self.number_of_sensors,
                })
                await asyncio.sleep(0.001)
