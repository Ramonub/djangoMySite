from django.urls import path
from . import views

urlpatterns = [
    path("index/", views.index, name = "index"),
    path("home/", views.home, name = "home"),
    path("chart/", views.test, name = "test"),
    path("standard/", views.standard, name = "standard"),
    path("chart/<str:room_name>/", views.room, name = "graph"),
    # path("", views.test, name='test'),
    # path("data", views.data, name = "data"),
]
