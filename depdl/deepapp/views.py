from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import torchvision
import torch
import torchvision.transforms as tf
import datetime

from django.views.decorators.csrf import csrf_exempt
from .models import Image


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


@csrf_exempt
def upload_image(request):

    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            image_object = Image(image=image_file)
            image_object.save()
            image_url = image_object.image.url
            return render(request, 'C:\\Work\\EmbryoVision\\EmbryoVision\\depdl\\deepapp\\templates\\image_work.html', {'image_url': image_url})
    return render(request, 'C:\\Work\\EmbryoVision\\EmbryoVision\\depdl\\deepapp\\templates\\image_work.html')


@csrf_exempt
def start_neural_network(request):
    model = torch.load('C:\\Work\\EmbryoVision\\data\\torch_type\\saved_model')
    model.eval()

    if request.method == 'POST':
        return JsonResponse({})


@csrf_exempt
def put_labels(request):
    if request.method == 'POST':
        return JsonResponse({'message': "Labels put on the image", 'labels': ""})
