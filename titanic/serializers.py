from rest_framework import serializers
from . models import titanic_guess

class titanic_guessSerializers(serializers.ModelSerializer):
    class Meta:
        model = titanic_guess
        fields='__all__'