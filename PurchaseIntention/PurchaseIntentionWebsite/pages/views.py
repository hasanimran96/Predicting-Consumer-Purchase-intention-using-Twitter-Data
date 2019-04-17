from django.shortcuts import render

def home(request):
    return render(request, "home.html", {})

def dashboard(request):
    return render(request, "dashboard.html", {})

def upload(request):
    return render(request, "upload.html", {})

def analysis(request):
    return render(request, "analysis.html", {})

def result(request):
    return render(request, "result.html", {})

    