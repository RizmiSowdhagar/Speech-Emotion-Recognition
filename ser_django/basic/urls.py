from django.urls import path
from . import views

urlpatterns = [
    path('pred/', views.pred, name='index'),
    path('', views.Voice_rec, name='record')
]