from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import torchvision
import torch
from visxd import cnn_model
from PIL import Image as plk
import torchvision.transforms as tf
import datetime


from django.views.decorators.csrf import csrf_exempt
from .models import Image

model = cnn_model.EmbryoModel()
model.load_state_dict(torch.load('C:\\Work\\EmbryoVision\\data\\torch_type\\final_cnn_model'))
model.eval()

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


@csrf_exempt
def upload_image(request):

    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            image_object = Image(image=image_file)
            image_object.save()
            request.session['image_id'] = image_object.id
            image_url = image_object.image.url

            return render(request, 'C:\\Work\\EmbryoVision\\EmbryoVision\\depdl\\deepapp\\templates\\image_work.html', {'image_url': image_url})
    return render(request, 'C:\\Work\\EmbryoVision\\EmbryoVision\\depdl\\deepapp\\templates\\image_work.html')





@csrf_exempt
def put_labels(request):
    if request.method == 'POST':
        image_id = request.session.get('image_id')
        if image_id is None:
            return JsonResponse({'error': 'Image ID NOT found in session'}, status=400)
        image_object = Image.objects.get(id=image_id)

        #im_obj = Image.objects.get(id=im_id)
        #img = plk.open(im_obj)
        #with torch.no_grad:
       #     preds = model(im_obj)

        return JsonResponse({'message': "Labels put on the image", 'labels': image_object})
    else:
        return JsonResponse({'error': 'Invalid requeST method'}, status=400)


@csrf_exempt
def start_neural_network(request):
    if request.method == 'POST':
        return JsonResponse({})
