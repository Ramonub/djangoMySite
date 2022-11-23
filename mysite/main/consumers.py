from __future__ import print_function
import json
from time import time
import asyncio
from tkinter.messagebox import showinfo

from channels.generic.websocket import WebsocketConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.generic.websocket import AsyncConsumer
from asgiref.sync import sync_to_async
from main import sensoren

class WSConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print('!!! Consumer connected !!!')
        await self.accept()
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'main_%s' % str(self.room_name)
        self.logging = False
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.channel_layer.send('sensor_stream', {'type': 'activate', 'group_name': self.room_group_name})

    async def disconnect(self, close_code):
        print('!!! Consumer disconnected !!!')
        await self.channel_layer.send('sensor_stream', {'type': 'deactivate'})
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

    
class WSManager(AsyncConsumer):
    def __init__(self, *args, **kwargs) -> None:
        self.numberOfMessage = 1
        self.startTime = time()
        AsyncConsumer.__init__(self, *args, **kwargs)

    async def activate(self, event):
        print("<<< Manager started >>>")
        self.sensors = sensoren.sensoren([0, 1])
        self.sensors.startScan()
        asyncio.create_task(self.run_sensor(self.sensors, event.get('group_name')))   # Create task with both sensors

    async def deactivate(self, event):
        self.sensors.stopScanning()
        print("<<< Manager stopped >>>")
    
    async def run_sensor(self, sensors, group_name):
        # print("Task started in group:", group_name)
        self.room_group_name = group_name
        self.running = True
        self.ftValue = 0
        self.dsValue = 0

        while self.running:
            try:
                self.sensorsValues = sensors.getValue()
            except:
                pass
            try:
                self.ftValue = self.sensorsValues[0]
                self.dsValue = self.sensorsValues[1]
            except:
                pass

            self.numberOfMessage += 1
            await self.channel_layer.group_send(self.room_group_name, {
                'type': 'chat_message',
                'dsValue': self.ftValue,
                'fsValue': self.dsValue,
                'messageNumber': self.numberOfMessage,
                'time': time() - self.startTime,
            })
            await asyncio.sleep(0.001)
