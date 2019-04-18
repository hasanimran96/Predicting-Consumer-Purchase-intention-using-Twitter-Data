from django.shortcuts import render
import csv, io
def home(request):
    return render(request, "home.html", {})

def dashboard(request):
    return render(request, "dashboard.html", {})

def upload(request):
    template = "upload.html"
    if request.method == "GET":
        return render(request, template,{})
     
    csv_file = request.FILES['file']
     
    if not csv_file.name.endswith('.csv'):
         messages.error(request,"This is not csv file")

    dataset = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(dataset)
    next(io_string)

    with open('uploadeddata/upload1.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv.reader(io_string, delimiter=',',quotechar="|"):
            filewriter.writerow(row)
    return render(request, template, {})

def analysis(request):
    return render(request, "analysis.html", {})

def result(request):
    return render(request, "result.html", {})

    