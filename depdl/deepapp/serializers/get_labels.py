from rest_framework.serializers import (
    Serializer,
    ListSerializer,
    IntegerField,
    FloatField,
)


class GetLabelsSerializer(Serializer):
    classification_pred = ListSerializer(child=IntegerField())
    regression_pred = ListSerializer(child=ListSerializer(child=FloatField()))
    hole_pred = IntegerField()

    class Meta:
        fields = (
            "classification_pred",
            "regression_pred",
            "hole_pred",
        )
