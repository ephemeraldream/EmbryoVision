from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import torch
from PIL import Image as PLImage
from torchvision.transforms.functional import pil_to_tensor


from django.views.decorators.csrf import csrf_exempt

from deepapp import cnn_model
from .models import Image
from pathlib import Path

model = cnn_model.EmbryoModel()
model.load_state_dict(torch.load(Path(__file__).resolve().parent / "final_cnn_model"))
model.eval()


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


@csrf_exempt
def upload_image(request):
    if request.method == "POST":
        image_file = request.FILES.get("image")
        if image_file:
            image_object = Image(image=image_file)
            image_object.save()
            # request.session['image_id'] = image_object.id
            image_url = image_object.image.url

            return render(
                request,
                "C:\\Work\\EmbryoVision\\EmbryoVision\\depdl\\deepapp\\templates\\image_work.html",
                {"image_url": image_url, "image_id": image_object.id},
            )
    return render(
        request,
        "C:\\Work\\EmbryoVision\\EmbryoVision\\depdl\\deepapp\\templates\\image_work.html",
    )


@csrf_exempt
def put_labels(request):
    if request.method == "POST":
        image_object = Image.objects.last()
        if image_object is None:
            return JsonResponse({"error": "Image ID NOT found in session"}, status=400)
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
        hole_pred = torch.ge(hole_pred, 0.5)
        gas_result = (
            "Gas outside of the holes detected and probably covers some part of the image."
            if hole_pred == 1
            else "Gas outside of the holes is not Detected. The image is clean."
        )

        return JsonResponse(
            {
                "message": "Labels put on the image",
                "classification_pred": cls_pred,
                "regression_pred": reg_pred,
                "hole_pred": gas_result,
            }
        )
    else:
        return JsonResponse({"error": "Invalid requeST method"}, status=400)


@csrf_exempt
def start_neural_network(request):
    if request.method == "POST":
        return JsonResponse({})
