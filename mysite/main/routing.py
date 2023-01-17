from django.urls import re_path, path

from .consumers import WSConsumer

ws_urlpatterns = [
    re_path('ws/show/', WSConsumer.as_asgi()),
]
