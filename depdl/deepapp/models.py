from django.db import models
from django.utils import timezone

# Create your models here.


class Question(models.Model):
    question_text = models.CharField(max_length=2000)
    pub_date = models.DateTimeField("date published")

    def __str__(self):
        return self.question_text


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=2000)
    votes = models.IntegerField(default=0)


class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    objects = models.Manager()


