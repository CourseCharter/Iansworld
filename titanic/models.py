from django.db import models

# Create your models here.
class titanic_guess(models.Model):

    class Gender(models.IntegerChoices):
        FEMALE = 0
        MALE = 1
    class Pclass(models.IntegerChoices):
        FIRST = 1
        SECOND = 2
        THIRD = 3

    sex=models.IntegerField(choices=Gender.choices)
    pclass=models.IntegerField(choices=Pclass.choices)
    age=models.IntegerField(default=0)
    relatives=models.IntegerField(default=0)
    fare=models.FloatField(default=0)
    result=models.IntegerField(default=0)

    def __str__(self):
        return super().__str__()