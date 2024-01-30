from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import AllowAny

from deepapp.models import Image
from deepapp.serializers.image import ImageSerializer


class ImageViewSet(ModelViewSet):
    queryset = Image.objects.all()
    permission_classes = [AllowAny]
    serializer_class = ImageSerializer
