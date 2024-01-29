from django.shortcuts import render
from django.http import HttpResponse
import datetime


def index(request):
    return HttpResponse(str(datetime.datetime.now()))
