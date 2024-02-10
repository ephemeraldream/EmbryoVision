import json
from django.contrib import admin
from deepapp import cnn_model
import torch
from pathlib import Path


from PIL import Image as PLImage
from torchvision.transforms.functional import pil_to_tensor


from deepapp.models import Image
from django.db.models import QuerySet
# Register your models here.

model = cnn_model.EmbryoModel()
model.load_state_dict(torch.load(Path(__file__).resolve().parent / "final_cnn_model"))
model.eval()


@admin.action(description="Generate labels")
def gen_labels(modeladmin, request, queryset: QuerySet):
    for image in queryset:
        processeed = PLImage.open(image.image)
        tensor = pil_to_tensor(processeed) / 255
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

        image.results = json.dumps(
            {
                "classification_pred": cls_pred,
                "regression_pred": reg_pred,
                "hole_pred": hole_pred,
            }
        )
        image.save()


class ImageAdmin(admin.ModelAdmin):
    list_display = ("id", "image")
    actions = [gen_labels]


admin.site.register(Image, ImageAdmin)
