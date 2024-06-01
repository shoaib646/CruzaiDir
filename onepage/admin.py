from django.contrib import admin
from .models import  Enquiry
# if no such table error python manage.py migrate --run-syncdb 


admin.site.register(Enquiry)
