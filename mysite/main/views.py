from django.http import HttpResponse
from django.shortcuts import render 

# Create your views here.
# dev_list = ul.get_daq_device_inventory(InterfaceType.ANY)
# daq_dev = ul.create_daq_device(0, dev_list[0])

def test(response):
    return render(response, 'main/chart.html')

def index(response):
    return render(response, "main/index.html", {})

def home(response):
    return render(response, 'main/home.html', {})

def chart(response):
    return render(response, 'main/home.html', {})

def standard(response):
    return render(response, 'main/chart.html', {})

def room(response, room_name):
    return render(response, 'main/chart.html', {
        'room_name': room_name
    })

# def data(response):
#     ds = DistanceSensor(0,0)
#     context = {
#         "sensorData": ds.getValue(),
#     }
#     return render(response, 'main/base.html', context)

# class DistanceSensor:
#     def __init__(self, board_num, channel) -> None:
#         self.board_num = board_num
#         self.channel = channel
#         self.x_values = []
#         self.y_values = []
#         self.ai_range = ULRange.BIP10VOLTS
#         print("Setting up sensor:", self.channel, "succeed.")

#     def getValue(self) -> float:
#         return (ul.to_eng_units(self.board_num, self.ai_range, ul.a_in(self.board_num, self.channel, self.ai_range)))




# from main.models import Dreamreal

# def crudops(request):
#     dreamreal = Dreamreal(
#         website = "www.polo.com", mail = "sorex@polo.com",
#         name = "sorex", phonenumber = "002376970"
#     )

#     dreamreal.save()