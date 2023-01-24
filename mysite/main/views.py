from django.shortcuts import render 

def show(response):
    return render(response, 'main/chart.html', {})

def logs(response):
    return render(response, 'main/logs.html', {})
