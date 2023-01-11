from django.urls import path
from . import views
from django.views.generic import TemplateView

app_name = 'user_datas'
urlpatterns = [
    path('survey/', views.registerSuervey, name='survey'),
    # path('', views.home, name='index'),
]
