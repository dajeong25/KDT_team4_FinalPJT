from django.contrib.auth.forms import UserCreationForm
from .models import User


class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = User
        fields = ['user_name', 'user_gender', 'user_age', 'user_email',
                  'pos_category', 'neg_category', 'allergy', 'category']

    def __init__(self, *args, **kwargs):
        super(CustomUserCreationForm, self).__init__(*args, **kwargs)
        self.fields['user_name'].widget.attrs['placeholder'] = "어떻게 불러드릴까요?(20자 이내)"
        # self.fields['user_gender'].widget.attrs['disabled'] = "선택"
        # self.fields['user_age'].widget.attrs['disabled'] = "선택"
        self.fields['user_email'].widget.attrs['placeholder'] = "이메일을 입력해주세요"

