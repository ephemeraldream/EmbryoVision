from django.urls import path
from . import views


app_name = 'deepapp'

urlpatterns = [
    path("xd", views.index, name="index"),
    path("", views.upload_image, name='upload_image')
]