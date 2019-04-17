from django.urls import path
from . import views
urlpatterns = [
    path('',views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload, name='upload'),
    path('analysis/', views.analysis, name='analysis'),
    path('result/', views.result, name='result'),
]
