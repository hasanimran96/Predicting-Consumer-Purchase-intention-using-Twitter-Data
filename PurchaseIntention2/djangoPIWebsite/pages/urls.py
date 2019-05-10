from django.urls import path
from . import views
urlpatterns = [
    path('',views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('annotate/', views.upload_annotate, name='annotate'),
    path('test/', views.upload_test, name='test'),
    path('analysis/', views.analysis, name='analysis'),
    path('result/', views.result, name='result'),
    path('testresult/', views.testresult, name='testresult'),
]
