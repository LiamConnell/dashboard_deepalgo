from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^runtest$', views.runtest, name='runningtest'),
]
