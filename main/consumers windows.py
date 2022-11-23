import json
from time import time
import asyncio
from tkinter.messagebox import showinfo

from .models import Sensor
from mcculw.enums import InterfaceType
from mcculw import ul

from channels.generic.websocket import WebsocketConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.generic.websocket import AsyncConsumer
from asgiref.sync import sync_to_async

class WSConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print('Consumer started')
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'main_%s' % str(self.room_name)
        self.logging = False
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.channel_layer.send('sensor_stream', {'type': 'activate', 'group_name': self.room_group_name})
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        try:
            if text_data_json['logging'] == 'True':
                self.logging = True
            else:
                self.logging = False
        except:
            pass
        print(self.logging)

    async def chat_message(self, event):

        await self.send(text_data=json.dumps({
            'dsValue': event['dsValue'],
            'fsValue': event['fsValue'],
            'messageNumber': event['messageNumber'],
            'time': event['time'],
        }))

        
        # dsValue = event['dsValue']
        # fsValue = event['fsValue']
        # messageNumber = event['messageNumber']
        # time = event['time']

    
class WSManager(AsyncConsumer):

    def __init__(self, *args, **kwargs) -> None:
        print("Manager started")
        self.numberOfMessage = 1
        self.startTime = time()
        AsyncConsumer.__init__(self, *args, **kwargs)

    async def activate(self, event):
        # Setup sensors
        try:
            dev_list = ul.get_daq_device_inventory(InterfaceType.ANY)
            daq_dev = ul.create_daq_device(0, dev_list[0])
        except:
            pass
        dsObject = await sync_to_async(Sensor.objects.get)(name='DistanceSensor')           # Set distance sensor from model
        fsObject = await sync_to_async(Sensor.objects.get)(name='forceSensor')              # Set force sensor from model
        asyncio.create_task(self.run_sensor(dsObject, fsObject, event.get('group_name')))   # Create task with both sensors

    async def deactivate(self, event):
        pass
    
    async def getValue(self, dsObj) -> float:
        return (ul.to_eng_units(dsObj.boardNumber, dsObj.ai_range, ul.a_in(dsObj.boardNumber, dsObj.channel, dsObj.ai_range)))

    async def run_sensor(self, dsObj: Sensor, fsObj: Sensor, group_name):
        print("Task started in group:", group_name)
        self.room_group_name = group_name
        self.running = True

        while self.running:
            self.numberOfMessage += 1
            await self.channel_layer.group_send(self.room_group_name, {
                'type': 'chat_message',
                'dsValue': await self.getValue(dsObj),
                'fsValue': await self.getValue(fsObj),
                'messageNumber': self.numberOfMessage,
                'time': time() - self.startTime,
            })
            await asyncio.sleep(0.01)
