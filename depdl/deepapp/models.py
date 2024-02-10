from django.db import models


class Image(models.Model):
    image = models.ImageField(upload_to="images/")
    results = models.TextField(blank=True, null=True)
