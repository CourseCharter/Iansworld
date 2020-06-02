from django.views.generic import TemplateView, View
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from django.contrib import messages
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from . forms import TitanicForm
from . import functions
import os
import pickle
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load


# Create your views here.
def Titanic_Home(request):
	form=TitanicForm()
	return render(request,'titanic/titanic_home.html', {'form': form})


def result(request):
		if request.method =='POST':
			form=TitanicForm(request.POST)
			if form.is_valid():
				sex=request.POST.get('sex')
				pclass=request.POST.get('pclass')
				age=request.POST.get('age')
				relatives=request.POST.get('relatives')
				fare=request.POST.get('fare')

				context = {
					'sex': sex,
					'pclass': pclass,
					'age': age,
					'relatives': relatives,
					'fare': fare,
				}
				#create Dataframe
				df=pd.DataFrame(context, index=[0])
				#send df to model
				titanic_mdl_location = os.path.join('titanic', 'titanic_model.pkl')
				titanic_sc_location = os.path.join('titanic', 'scalars.pkl')
				mdl=load(titanic_mdl_location)
				sc=load(titanic_sc_location)
				# mdl=load("/Users/user/projects/iansworld/titanic/titanic_model.pkl")
				# sc=load("/Users/user/projects/iansworld/titanic/scalars.pkl")
				df=sc.transform(df)
				y_pred=mdl.predict_classes(df)
				y_pred=pd.DataFrame(y_pred, columns=['Survived'])
				y_pred=y_pred.replace({1:'Lived', 0:'Perished'})
				form=TitanicForm()

			return render(request, 'titanic/result.html', {'pred': y_pred.iloc[0]['Survived']})
		
		else:
			form=TitanicForm()

		return render(request, 'titanic/titanic_home.html', {'form': form})

# class Titanic_Guess_View(viewsets.ModelViewSet):
# 	queryset = titanic_guess.objects.all()
# 	serializer_class = titanic_guessSerializers
		


# def titanic_page_guess(request):

# 	if request.method =='POST':
# 		form=TitanicForm(request.POST)
# 		if form.is_valid():
# 			sex=form.cleaned_data['sex']
# 			pclass=form.cleaned_data['pclass']
# 			age=form.cleaned_data['age']
# 			relatives=form.cleaned_data['relatives']
# 			fare=form.cleaned_data['fare']
# 			myDict = (request.POST).dict()
# 			df=pd.DataFrame(myDict, index=[0])
# 			df=df.drop(['csrfmiddlewaretoken'], axis=1)
# 			answer = functions.survived(df)
# 			print(answer)
# 			#messages.success(request, 'Guess Submitted.')
# 			#messages.success(request, 'Your Fate: {}'.format(answer))
# 			#this is causing a problem for multiple attempts. the guess is being submitted multiple times but the response cannot be rendered multiple times
# 			#titanic_table=create_HTML_table(titanic_df(pd.read_csv('titanic/data.csv')))
# 			context = {
# 				'answer': answer,
# 			}
# 			return render(request, 'titanic/testform.html', {'context': context})

# 	else:
# 		form=TitanicForm()

# 	return render(request, 'titanic/testform.html', {'form': form}) 

