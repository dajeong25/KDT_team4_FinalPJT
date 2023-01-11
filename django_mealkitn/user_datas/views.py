import json
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import generic
from .forms import CustomUserCreationForm
from .models import User


def registerSuervey(request):
    form = CustomUserCreationForm()
    allergy_list = []
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        allergy_list = request.POST.getlist('jobb')
        if form.is_valid():
            form.save(commit=False)
            return render(request, 'mk_recs/product.html', )
    context = {'form': form, 'allergy_list': allergy_list}
    return render(request, 'survey.html', context)

