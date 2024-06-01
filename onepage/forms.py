from django import forms
from .models import Enquiry
# , OptimalDB, Flowdb
# from Capstone.TrafficData.TrafficFlowPredictor import *


class enquiryform(forms.ModelForm):

    class Meta:

        model = Enquiry
        fields = "__all__"

        widgets = {
            "name": forms.TextInput(attrs={"placeholder":"Name","class":"form-control"}),
            "email_or_contact": forms.TextInput(attrs={ "placeholder":"Email or Contact Number","class":"form-control"}),
            "subject": forms.TextInput(attrs={"placeholder":"Subject","class":"form-control"}),
            "message" : forms.Textarea(attrs={"placeholder":"Message", "class":"form-control"}),
        }


''' Real Deployment
class OptimalForm(forms.ModelForm):
    class Meta:
        model = OptimalDB
        fields = '__all__'

        widgets = {
            'Select_Model': forms.Select(attrs={'class': 'form-control'}),
            'Source': forms.NumberInput(attrs={'class': 'form-control','placeholder': 'e.g 6145'}),
            'Destination': forms.NumberInput(attrs={'class': 'form-control','placeholder': 'e.g 9732'}),
            'Date_Time': forms.DateTimeInput(attrs={'class': 'form-control'}),
        }

class Flowform(forms.ModelForm):
    class Meta:
        model = Flowdb
        fields = '__all__'

        widgets = {
            'Forecast_Route_Traffic': forms.Select(attrs={'class': 'form-control'}),
            'Route': forms.NumberInput(attrs={'class': 'form-control','placeholder': 'e.g 9732'}),
        }
'''