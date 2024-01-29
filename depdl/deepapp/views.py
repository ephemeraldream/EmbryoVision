from django.shortcuts import render
from django.http import HttpResponse
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
    if request.method == 'POST':
        return HttpResponse("Neural network started.")


@csrf_exempt
def put_labels(request):
    if request.method == 'POST':
        return HttpResponse("Labels put on the image.")
