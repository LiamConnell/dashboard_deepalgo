from django.shortcuts import render
from django.http import HttpResponse

sys.path.insert(1,os.path.join(sys.path[0], '../TFcode'))
import TF_codemonster

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def runtest(request):
    trainreturn_, testreturn_ = TF_codemonster.run()
    response = ' '.join(( str(trainreturn_), str(testreturn_) ))
    return HttpResponse("Hello, world. You're at the polls index.")