from django.urls import path

from deepapp.views.image import ImageViewSet
from deepapp.views.gen_labels import gen_labels
from rest_framework.routers import SimpleRouter


app_name = "deepapp"

router = SimpleRouter()
router.register("image", ImageViewSet, basename="image")

urlpatterns = [
    path("gen_labels/<int:id>", gen_labels, name="put_labels"),
]

urlpatterns += router.urls
