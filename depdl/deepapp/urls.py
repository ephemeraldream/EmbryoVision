from django.urls import path
from . import views


app_name = 'deepapp'

urlpatterns = [
    path("xd", views.index, name="index"),
    path("", views.upload_image, name='upload_image'),
    path("start_nn/", views.start_neural_network, name='start_neural_network'),
    path("put_labels/", views.put_labels, name='put_labels')
]
