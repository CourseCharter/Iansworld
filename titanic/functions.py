# from joblib import load,dump
# import pandas as pd
# import numpy as np
# import pickle
# import json
# from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
# from django.shortcuts import render
# from rest_framework import viewsets
# from rest_framework.decorators import api_view
# from django.core import serializers
# from django.contrib import messages
# from rest_framework.response import Response
# from rest_framework import status
# from django.http import JsonResponse
# import warnings
# warnings.filterwarnings("ignore")
# np.random.seed(42)


# def load_model():
#     model = load('titanic_model.pkl')
#     print("Model Loaded!")
#     return model

# def titanic_df(titanic_data):
			
# 	#cleaning the data
# 	titanic_data.replace('?', np.nan, inplace= True)
# 	titanic_data.replace({'male': 1, 'female': 0}, inplace=True)
# 	titanic_data = titanic_data.astype({"age": np.float64, "fare": np.float64})

# 	#combining siblings and parents into relatives
# 	titanic_data['relatives'] = titanic_data.apply (lambda row: int((row['sibsp'] + row['parch']) > 0), axis=1)
			
# 	titanic_df=pd.DataFrame(titanic_data)
			
# 	return titanic_df

#@api_view(["POST"])
# def survived(df):
#     print(df)
#     mdl=load("/Users/user/projects/iansworld/titanic/titanic_model.pkl")
#     sc=load("/Users/user/projects/iansworld/titanic/scalars.pkl")
#     df=sc.transform(df)
#     print(df)
#     y_pred=mdl.predict_classes(df)
#     return y_pred