from django.shortcuts import render
from django.contrib import messages
import csv, io
def home(request):
    return render(request, "home.html", {})

def dashboard(request):
    return render(request, "dashboard.html", {})
   
def upload_annotate(request):
        template = "annotate.html"
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

                # with open('Data.csv', newline='', encoding='utf-8-sig') as csvFile:
                # for line in csv.reader(csvFile, delimiter=';'):


                with open('uploadeddata/'+filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        for row in csv.reader(io_string, delimiter=',',quotechar="|"):
                                filewriter.writerow(row)

                copied = 1            
                return render(request, template, {"success":copied})


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

                # with open('Data.csv', newline='', encoding='utf-8-sig') as csvFile:
                # for line in csv.reader(csvFile, delimiter=';'):


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
    return render(request, "result.html", {})

    