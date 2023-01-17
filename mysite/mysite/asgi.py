'''
ASGI config for mysite project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
'''

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django_asgi_app = get_asgi_application()

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter
from channels.routing import URLRouter, ChannelNameRouter
from main.routing import ws_urlpatterns
from main import consumers
# import django

# django.setup()

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': URLRouter(ws_urlpatterns),
    'channel': ChannelNameRouter({
        'sensor_stream': consumers.WSManager.as_asgi()
    })
})
