import json
from time import time
from . import views
import asyncio
import subprocess

from channels.generic.websocket import WebsocketConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.generic.websocket import AsyncConsumer
from asgiref.sync import async_to_sync

class WSConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'main_%s' % str(self.room_name)

        # Starting background worker
        # path = r"C:\Users\ramon\OneDrive\Documenten\Ramon\Stage_madern\Documenten\Django\mysite\manage.py"
        # commando = "pythonw "+ path +" runworker sensor_stream"
        # self.i = subprocess.Popen(commando, creationflags=subprocess.CREATE_NEW_CONSOLE, shell=False)

        # print(self.room_group_name)
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        # await self.sendStartMessage()
        await self.accept()
        print('after accepting')

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        # Terminate background worker
        self.i.terminate()

    async def sendStartMessage(self, event):
        print("sending message")
        self.numberOfValues = event.get('numberOfValues')
        await self.channel_layer.send('sensor_stream', {'type': 'activate', 'sensorCode': 0, 'group_name': self.room_group_name, 'numberOfValues': self.numberOfValues})

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        self.methodName = text_data_json['type']
        self.numberOfValues = text_data_json['numberOfValues']
        print("Message recieved")
        print(self.methodName)
    
        await self.channel_layer.send('sensor_stream', {'type': self.methodName, 'numberOfValues': self.numberOfValues, 'group_name': self.room_group_name})
        print('recieve')

    async def chat_message(self, event):
        sensorValue = event['sensorValue']
        messageNumber = event['messageNumber']
        time = event['time']

        await self.send(text_data=json.dumps({
            'sensorValue': sensorValue,
            'messageNumber': messageNumber,
            'time': time
        }))

    async def test(self):
        pass

    
class WSManager(AsyncConsumer):

    def __init__(self, *args, **kwargs):
        print("Manager accessable")
        self.ds = views.DistanceSensor(0, 0)
        AsyncConsumer.__init__(self, *args, **kwargs)

    async def activate(self, event):
        print("Manager activated")
        asyncio.create_task(self.run_sensor(event))

    async def secondActivate(self, event):
        print("other Manager activated")
        asyncio.create_task(self.runMessages(event))

    async def deactivate(self, event):
        pass

    async def runMessages(self, dsObj):
        self.room_group_name = dsObj.get('group_name')
        # self.callableMethod = dsObj.get('type')
        self.numberOfValues = dsObj.get('numberOfValues')
        print("run start message wiht value:", self.numberOfValues)
        await self.channel_layer.group_send(self.room_group_name, {
            'type': 'sendStartMessage',
            'numberOfValues': self.numberOfValues
        })

    async def run_sensor(self, dsObj):
        self.startTime = time()
        self.room_group_name = dsObj.get('group_name')
        self.numberOfValues = dsObj.get('numberOfValues')
        if self.numberOfValues == 'x':
            self.numberOfValues = int(2e+20)
        else:
            try:
                self.numberOfValues = int(self.numberOfValues)
            except:
                self.numberOfValues = 0

        print(self.numberOfValues)
            
        print("Running", self.numberOfValues, "times")
        # while sensor in running plot update
        for i in range (int(self.numberOfValues)):
            # print("number:", i)
            await self.channel_layer.group_send(self.room_group_name, {
                'type': 'chat_message',
                'sensorValue': self.ds.getValue(),
                'messageNumber': i,
                'time': time() - self.startTime
            })
            await asyncio.sleep(0.01)
        print("done")
