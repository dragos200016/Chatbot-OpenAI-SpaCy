from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('chat/', views.chat, name='chat'),
    path('', views.index, name='index'),
    path('delete_chat_history/', views.delete_chat_history, name='delete_chat_history'),
]
