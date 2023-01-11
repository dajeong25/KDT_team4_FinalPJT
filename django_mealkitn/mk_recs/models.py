from django.db import models


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
