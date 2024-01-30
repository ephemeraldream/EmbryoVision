from rest_framework.serializers import ModelSerializer

from deepapp.models import Image


class ImageSerializer(ModelSerializer):
    class Meta:
        model = Image
        fields = (
            "id",
            "image",
        )
