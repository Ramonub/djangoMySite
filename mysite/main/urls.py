from django.urls import path
from . import views
from mysite import settings
from django.conf.urls.static import static

urlpatterns = [
    path("show/", views.show, name = "room"),
    path("logs/", views.logs, name = "logs"),
] 

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) 
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
