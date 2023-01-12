import json
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views import generic
from .forms import UserCreationForm
from .models import User
from mk_recs.models import coupang_category, CoupangMealkit


def registerSuervey(request, *args, **kwargs):
    form = UserCreationForm()
    # 선호, 비선호 조사 카테고리 추출
    categorys = coupang_category.objects.values_list('category_keys', flat=True).distinct()
    
    # 대분류, 소분류 항목 추출
    category_1 = CoupangMealkit.objects.values_list('category_name', flat=True).distinct()
    category_dict = {}
    for category in category_1:
        category_2 = CoupangMealkit.objects.filter(category_name=category).values_list('category_second', flat=True).distinct()
        category_dict[category] = category_2
    print(category_dict)

    if request.method == 'POST':
        form = UserCreationForm()
        form.user_name = request.POST['user_name']
        form.user_gender = request.POST['user_gender']
        form.user_age = request.POST['user_age']
        form.user_email = request.POST['user_email']
        form.allergy_list = request.POST.getlist('jobb')
        form.pos_list = request.POST.getlist('pos_category[]')
        form.neg_list = request.POST.getlist('neg_category[]')
        form.category = request.POST['category_fst']
        form.category_sec = request.POST.getlist('category_sec')
        print(form)
        form.save()
        if form.is_valid():
            print('문제?')
            form.save()
            print('form 저장')
            return redirect('mk_recs_result')
    context = {'form': form, 'category': categorys, 'category_dict': category_dict}
    print('다시')
    return render(request, 'survey.html', context)

