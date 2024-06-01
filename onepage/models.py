from django.db import models
from django import forms
from django.core.validators import RegexValidator
# from Capstone.TrafficData.TrafficFlowPredictor import *


class Enquiry(models.Model):

    name = models.CharField(max_length=265)
    email_or_contact = models.CharField(max_length=265,validators=[
        RegexValidator(
            regex=r'^(\d{10}|\w+@\w+\.\w{2,3})$',
            message='Please enter a valid email or contact number.'
        )
    ])
    subject = models.CharField(max_length=265, blank=True)
    message = models.CharField(max_length=265, blank=True)

    def __str__(self):
        return str(f'{str(self.name)}:{str(self.email_or_contact)}')


'''Real  Deployment
# class OptimalDB(models.Model):
#     Select_Model = models.CharField(max_length=20, choices=[(e.value, e.name) for e in TrafficFlowModelsEnum], default=TrafficFlowModelsEnum.SELECT.value)
#     Source = models.IntegerField(null=False, blank=False)
#     Destination = models.IntegerField(null=False, blank=False)
#     Date_Time = models.DateTimeField(null=True, blank=True)

# class Flowdb(models.Model):
#     Forecast_Route_Traffic = models.CharField(max_length=20, choices=[(e.value, e.name) for e in TrafficFlowModelsEnum], default=TrafficFlowModelsEnum.SELECT.value)
#     Route = models.IntegerField(null=True, blank=True)

'''



    








