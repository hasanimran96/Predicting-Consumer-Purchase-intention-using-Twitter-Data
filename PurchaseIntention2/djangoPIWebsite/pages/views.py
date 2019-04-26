from django.shortcuts import render
from django.contrib import messages
import csv, io
import codecs
from django.http import HttpResponse
from matplotlib import pylab
from pylab import *
# from io import StringIO
import six
import PIL, PIL.Image
import numpy as np

def home(request):
    return render(request, "home.html", {})

def dashboard(request):
    return render(request, "dashboard.html", {})
   
def upload_annotate(request):
        from pages.StandardProcessing import  read_dir
        template = "annotate.html"
        if request.method == "GET":
                return render(request, template,{"files" : read_dir})
     
        csv_file = request.FILES['file']
        filename = csv_file.name

        if not csv_file.name.endswith('.csv'):
                messages.error(request,"This is not csv file")
                err = 1
                return render(request, template, {"error":err, "files" : read_dir})
                
        if csv_file.name.endswith('.csv'):
                dataset = csv_file.read().decode('utf-8','ignore')
                io_string = io.StringIO(dataset)       
                
                with open('uploadeddata/'+filename, 'w', encoding='utf-8') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        count = 0                        
                        for row in csv.reader(io_string, delimiter=',',quotechar="|"):
                                if count == 0:
                                        if (row[0].lower() =="class" or row[0].lower()== "text") and (row[1].lower() =="class" or row[1].lower()== "text"):  
                                                filewriter.writerow(row)
                                        else:
                                                err = 1          
                                                return render(request, template, {"error":err, "files" : read_dir})
                                elif count > 0:
                                     filewriter.writerow(row)   
                                count += 1                     
                copied = 1            
                return render(request, template, {"success":copied, "files" : read_dir})


def upload_test(request):
        template = "test.html"
        if request.method == "GET":
                return render(request, template,{})
     
        csv_file = request.FILES['file']
        filename = csv_file.name

        if not csv_file.name.endswith('.csv'):
                messages.error(request,"This is not csv file")
                err = 1
                return render(request, template, {"error":err})
                
        if csv_file.name.endswith('.csv'):
                dataset = csv_file.read().decode('UTF-8')
                io_string = io.StringIO(dataset)

                with open('uploadeddata/'+filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        for row in csv.reader(io_string, delimiter=',',quotechar="|"):
                                filewriter.writerow(row)

                copied = 1            
                return render(request, template, {"success":copied})


def analysis(request):    
    return render(request, "analysis.html", {})

def result(request):
    from pages.StandardProcessing import output_to_results
    if request.method == "POST": 
        filename = request.POST.get('file1')
        model = request.POST.get('model')
        docVector = request.POST.get('doc')
        # print(filename)
        path1 = "uploadeddata\\"+filename      
        return render(request, "result.html", {"out1": output_to_results(path1,docVector,model)
        , "out2": model})        
    

    