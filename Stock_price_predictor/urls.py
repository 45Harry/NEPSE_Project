from django.urls import path
from .views import stock_data

urlpatterns = [
    path('stock/', stock_data),
]