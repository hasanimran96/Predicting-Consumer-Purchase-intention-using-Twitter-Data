from django.shortcuts import render
from django.contrib import messages
import csv, io
import codecs
from django.http import HttpResponse
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
                dataset = csv_file.read().decode('ascii','ignore')
                io_string = io.StringIO(dataset)
                io_string1 = io.StringIO(dataset)
                temp_count = 0       
                row1 = csv.reader(io_string, delimiter=',',quotechar="|")
                flag_valid = 0
                for row_data in row1:
                        if temp_count == 0:
                                temp_list =  [x.lower() for x in row_data] 
                                if(set(["class","text"]).issubset(set(temp_list))): 
                                        flag_valid = 1
                                else:
                                        flag_valid = 0
                                break            


                if flag_valid == 1:
                        with open('uploadeddata/'+filename, 'w', encoding='ascii') as csvfile:
                                filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)                        
                                for row in csv.reader(io_string1, delimiter=',',quotechar="|"):
                                        filewriter.writerow(row)                                       
                                copied = 1            
                                return render(request, template, {"success":copied, "files" : read_dir})
                elif flag_valid == 0:
                        err = 1                
                        return render(request, template, {"error":err, "files" : read_dir})

def upload_test(request):
        from pages.StandardProcessing import  read_dir
        template = "test.html"
        if request.method == "GET":
                return render(request, template,{"files" : read_dir})
     
        csv_file = request.FILES['file']
        filename = csv_file.name

        if not csv_file.name.endswith('.csv'):
                messages.error(request,"This is not csv file")
                err = 1
                return render(request, template, {"error":err, "files" : read_dir})
                
        if csv_file.name.endswith('.csv'):
                dataset = csv_file.read().decode('ascii','ignore')
                io_string = io.StringIO(dataset)
                io_string1 = io.StringIO(dataset)
                temp_count = 0       
                row1 = csv.reader(io_string, delimiter=',',quotechar="|")
                flag_valid = 0
                for row_data in row1:
                        if temp_count == 0:
                                temp_list =  [x.lower() for x in row_data] 
                                if(set(["class","text"]).issubset(set(temp_list))): 
                                        flag_valid = 1
                                else:
                                        flag_valid = 0
                                break            


                if flag_valid == 1:
                        with open('uploadeddata/'+filename, 'w', encoding='ascii') as csvfile:
                                filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)                        
                                for row in csv.reader(io_string1, delimiter=',',quotechar="|"):
                                        filewriter.writerow(row)                                       
                                copied = 1            
                                return render(request, template, {"success":copied, "files" : read_dir})
                elif flag_valid == 0:
                        err = 1                
                        return render(request, template, {"error":err, "files" : read_dir})


def analysis(request):    
    from pages.data_analysis import output_to_analysis 
    class_count, frequent_words, negative_tweets_str, positive_tweets_str = output_to_analysis("uploadeddata\Annotated4.csv")
    return render(request, "analysis.html", {"count":class_count,
    "word":frequent_words, "neg":negative_tweets_str, "pos":positive_tweets_str})

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
    

def testresult(request):   
     from pages.ModelTest import output_to_results
     if request.method == "POST": 
        train = request.POST.get('file1')
        test = request.POST.get('file2')
        model = request.POST.get('model')
        docVector = request.POST.get('doc')
        level1 = (request.POST.get('level_1')).split("-")
        level2 = request.POST.get('level_2').split("-")
        level3 = request.POST.get('level_3').split("-")
        data_level1 = level1[0]
        data_level2 =  level2[0]
        data_level3 =  level3[0]
        # print(filename)
        trainpath = "uploadeddata\\"+train 
        testpath = "uploadeddata\\"+test   
        stats, test_data, potential_df, pie_data =  output_to_results(trainpath,testpath, docVector, model, data_level1, data_level2, data_level3)                   
        return render(request, "result_test.html", {"out1": test_data, "potential":potential_df,"pie":pie_data})
       