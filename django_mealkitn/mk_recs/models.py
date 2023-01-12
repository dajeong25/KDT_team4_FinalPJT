from django.db import models

# Create your models here.
class CoupangMealkit(models.Model):
    category_name = models.CharField(max_length=300, blank=True, null=True)
    category_second = models.CharField(max_length=300, blank=True, null=True)
    allergy = models.CharField(max_length=300, blank=True, null=True)
    product_id = models.CharField(max_length=300, blank=True, null=True)
    data_item_id = models.CharField(max_length=300, blank=True, null=True)
    data_vendor_item_id = models.CharField(max_length=300, blank=True, null=True)
    product = models.CharField(max_length=300, default='product', blank=True, primary_key=True)
    product_name = models.CharField(max_length=300, blank=True, null=True)
    full_price = models.IntegerField(blank=True, null=True)
    discounted_rate = models.CharField(max_length=300, blank=True, null=True)
    discounted_price = models.IntegerField(blank=True, null=True)
    yogiyo_price = models.IntegerField(blank=True, null=True)
    ratings = models.FloatField(blank=True, null=True)
    review_count = models.IntegerField(blank=True, null=True)
    issoldout = models.CharField(max_length=300, blank=True, null=True)
    ingredient = models.CharField(max_length=300, blank=True, null=True)
    edited_ingredient = models.CharField(max_length=300, blank=True, null=True)
    price_per_100g = models.CharField(max_length=300, blank=True, null=True)
    weighted_rating = models.FloatField(blank=True, null=True)

    def __str__(self):
        return self.product

    class Meta:
        db_table = 'CoupangMealkit'
        verbose_name = '쿠팡 밀키트 정보'  # 관리자페이지에 모델관리할 때 출력되는 모델이름
        verbose_name_plural = '쿠팡 밀키트 정보'  # 기본적으로 영어기준 복수로 모델명이 표시됨


# 쿠팡 세부카테고리 리스트 등록
class coupang_category(models.Model):
    category_keys = models.CharField(max_length=15)

    def __str__(self):
        return self.category_keys

    class Meta:
        db_table = 'coupang_category'
        verbose_name = '쿠팡 세부 카테고리'  # 관리자페이지에 모델관리할 때 출력되는 모델이름
        verbose_name_plural = '쿠팡 세부 카테고리'  # 기본적으로 영어기준 복수로 모델명이 표시됨
