from django.urls import path
from . import views

urlpatterns = [
    path('mk_recs/', views.mk_recs, name='mk_recs'),
]
