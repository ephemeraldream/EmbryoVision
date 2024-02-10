import torch

from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser
from rest_framework.decorators import api_view, parser_classes, permission_classes

from pathlib import Path
from PIL import Image as PLImage
from torchvision.transforms.functional import pil_to_tensor
from deepapp import cnn_model
from deepapp.serializers.get_labels import GetLabelsSerializer


from ..models import Image

from rest_framework.request import Request
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

model = cnn_model.EmbryoModel()
model.load_state_dict(
    torch.load(Path(__file__).resolve().parent.parent / "final_cnn_model")
)
model.eval()


@extend_schema(
    methods=["POST"],
    responses=GetLabelsSerializer,
)
@permission_classes([AllowAny])
@parser_classes([MultiPartParser])
@api_view(["POST"])
def gen_labels(_request: Request, id: int) -> Response:
    image_object = Image.objects.get(id=id)
    img = image_object.image
    processeed = PLImage.open(img)
    tensor = pil_to_tensor(processeed)
    tensor = tensor / 255
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        reg_pred, cls_pred, hole_pred = model(tensor)

    cls_pred = cls_pred.squeeze(0)
    reg_pred = reg_pred.squeeze(0)

    _, cls_pred = torch.max(cls_pred, dim=1)
    cls_pred = cls_pred.tolist()
    reg_pred = reg_pred * 255
    reg_pred = reg_pred.tolist()
    hole_pred = 1 if torch.ge(hole_pred, 0.5).item() is False else 0

    return Response(
        GetLabelsSerializer(
            {
                "classification_pred": cls_pred,
                "regression_pred": reg_pred,
                "hole_pred": hole_pred,
            }
        ).data
    )
