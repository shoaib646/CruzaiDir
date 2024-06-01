from django.urls import path
from .views import homepage,forminfo
# , OptimalView, view_map


urlpatterns = [
    path('', homepage, name='home'),
    path(r'result',forminfo, name='result'),
]

''' Real Deployment
    path(r'OptimalPath', OptimalView, name='optimalpath'),
    path(r'Map', view_map, name='map'),'''
