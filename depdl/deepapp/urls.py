from django.urls import path

from deepapp.views.image import ImageViewSet
from .views import views_test
from rest_framework.routers import SimpleRouter


app_name = "deepapp"

router = SimpleRouter()
router.register("image", ImageViewSet, basename="image")

urlpatterns = [
    path("xd", views_test.index, name="index"),
    path("", views_test.upload_image, name="upload_image"),
    path("start_nn/", views_test.start_neural_network, name="start_neural_network"),
    path("put_labels/", views_test.put_labels, name="put_labels"),
]


urlpatterns += router.urls
