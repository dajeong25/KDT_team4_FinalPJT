from django.urls import path
from . import views

app_name = 'mk_recs'
urlpatterns = [
    # path('index/', views.home, name='index'),
    path('mealkit/', views.mk_recs_result, name='mk_recs_result'),
    path('recipe/', views.rcp_recs_result, name='recipe'),
]
