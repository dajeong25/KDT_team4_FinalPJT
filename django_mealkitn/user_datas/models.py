from django.db import models


class User(models.Model):
    GENDERS = (
        ('1', '남성(Man)'),
        ('2', '여성(Woman)'),
    )
    AGE_GROUP = (
        ('~19세', '10대이하'),
        ('20~29세', '20대'),
        ('30~39세', '30대'),
        ('40~49세', '40대'),
        ('50~59세', '50대'),
        ('60세~', '60대이상'),
    )
    category1 = (
        ('1', '국탕전골'),
        ('2', '덮밥/비빔밥'),
        ('3', '스테이크/고기'),
        ('4', '면/파스타/감바스'),
        ('5', '분식'),
        ('6', '중식요리'),
        ('7', '기타요리'),
    )
    user_name = models.CharField(verbose_name='이름', max_length=20)
    user_gender = models.CharField(verbose_name='성별', max_length=6, choices=GENDERS, blank=True)
    user_age = models.CharField(verbose_name='연령대', max_length=10, choices=AGE_GROUP, blank=True)
    user_email = models.EmailField(verbose_name='이메일', max_length=50, blank=True)
    pos_category = models.CharField(verbose_name='pos_category', max_length=200, blank=True)
    neg_category = models.CharField(verbose_name='neg_category', max_length=200, blank=True)

    allergy = models.CharField(verbose_name='알레르기', max_length=100, blank=True)
    category = models.CharField(verbose_name='대분류', max_length=200, choices=category1)
    category_sec = models.CharField(verbose_name='소분류', max_length=200)

    def __str__(self):
        return self.user_name

    class Meta:
        db_table = 'Users'
        verbose_name = '유저 정보'  # 관리자페이지에 모델관리할 때 출력되는 모델이름
        verbose_name_plural = '유저 정보'  # 기본적으로 영어기준 복수로 모델명이 표시됨
