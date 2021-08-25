from django.urls import path

from . import views

from django.contrib.auth.views import LoginView,LogoutView
from django.urls import path, include 
from django.views.generic.base import TemplateView
from django.contrib.auth import views as auth_views

app_name = 'forecasting'
urlpatterns = [
	
	path('', views.IndexView.as_view(), name='index'),
	path('home/', include('django.contrib.auth.urls')), # new
	path('home/', views.home, name='home'),

	path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
]