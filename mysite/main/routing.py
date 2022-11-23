from django.urls import re_path, path

from .consumers import WSConsumer

ws_urlpatterns = [
    re_path('ws/chart/(?P<room_name>\w+)/$', WSConsumer.as_asgi()),
    re_path('ws/', WSConsumer.as_asgi()),
]