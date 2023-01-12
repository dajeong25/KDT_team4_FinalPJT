from django.urls import path
from user_datas.views import registerSuervey

app_name = 'user_datas'
urlpatterns = [
    path('survey/', registerSuervey, name='survey'),
]
