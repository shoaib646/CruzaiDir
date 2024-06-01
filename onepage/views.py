from django.shortcuts import render
from django.urls import reverse
import joblib, pickle
import pandas as pd
import numpy as np
from .forms import enquiryform
# , OptimalForm, Flowform
from django.contrib import messages
# from Capstone.TrafficData.TrafficFlowPredictor import *
# from Capstone  import route_finding as router
import datetime

def homepage(request):
    form = enquiryform()
    if request.method == 'POST':
        form = enquiryform(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Request submitted successfully.')
            form = enquiryform()
        else:
            messages.error(request, "Please enter a valid email or contact number.")
            form = enquiryform
            
    template_name = 'index.html'
    return render(request, template_name, context={'form': form})



def forminfo(request):
    value = request.GET['experience']
    input_value = int(value)

    if input_value == 0:
        newdata = pd.Series([int(input_value)])
        data_pred = pd.DataFrame(newdata, columns = ['YearsExperience'])
        v = int(data_pred.values[0][0])
        predicted_value = 'His/Her Salary range depends on skills and Qualification'
    else:
        newdata = pd.Series([int(input_value)])
        data_pred = pd.DataFrame(newdata, columns = ['YearsExperience'])
        v = int(data_pred.values[0][0])

        with open('./SavedModels/windowmodels_1.joblib', 'rb') as file:
            model = joblib.load(file)
            try:
                pred_value =  int(model.predict(data_pred).values[0])
            except:
                pred_value = int(model.predict(data_pred))

            lowest_value = str(pred_value)
            highest_value = str(pred_value + 10290)
            range_ = f"${lowest_value} to ${highest_value}"
            predicted_value = str(range_)

    return render(request, 'result.html', {'yoe':v, 'predict':predicted_value})



''' Real Deployment
# def OptimalView(request):
#     form = OptimalForm(prefix='optimal')
#     flow_form_= Flowform(prefix='flow')
#     routes = None
#     traffic = None
#     optimalform = OptimalForm()
#     flow_form = Flowform()
    
#     if request.method == 'POST':
#         if "submit_optimal" in request.POST:
#             optimalform = OptimalForm(request.POST)
#             if optimalform.is_valid():
#                 Selected_Model = optimalform.cleaned_data['Select_Model']
#                 Source = optimalform.cleaned_data['Source']
#                 Destination = optimalform.cleaned_data['Destination']
#                 Date_String = optimalform.cleaned_data['Date_Time']
#                 routes = router.runRouter(Source, Destination, parse_date(Date_String), Selected_Model)  
#                 optimalform.save()  
#                 optimalform = OptimalForm(prefix='optimal')

#         elif "submit_flow" in request.POST:
#             flow_form = Flowform(request.POST)
#             if flow_form.is_valid():
#                 model = flow_form.cleaned_data['Forecast_Route_Traffic']
#                 point = flow_form.cleaned_data['Route']

#                 predictor = TrafficFlowPredictor()
#                 date = parse_date(date_string=None)
#                 flow = predictor.predict_traffic_flow(point, date, 4, model)

#                 for key,value in dict(pd.read_csv('Capstone/data/traffic_network2.csv')[['Stop_IDs','Site Description']].values).items():
#                     if key == point:
#                         traffic = f"--Predicted Traffic Flow--\nRoute_ID:\t\t{value}\nTime:\t\t{date.strftime('%Y/%m/%d %I:%M:%S')}\nPrediction:\t{str(int(flow))} veh/hr"
#                         print(traffic)
#                         flow_form.save()
#                         flow_form = Flowform(prefix='flow')

#     template_name = 'TermI.html'
#     context = {'form': optimalform, 'routes': routes, 'flow_form':flow_form, 'traffic': traffic}
#     return render(request, template_name, context)








# def view_map(request):
#     file_path = 'D:/Porfolio/FreelanceWeb/onepage/templates/plotted.html'
#     return render(request, 'plotted.html', {'file_path': file_path})

# def parse_date(date_string):
#     try:
#         if date_string != None:
#             date = datetime.datetime.strptime(str(date_string), "%Y/%m/%d %H:%M:%S")
#         else:
#             date = datetime.datetime.now()
#     except ValueError:
#         date = datetime.datetime.now()
#     return date

'''
