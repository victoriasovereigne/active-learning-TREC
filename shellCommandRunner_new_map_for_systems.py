from scipy.stats.stats import kendalltau
from numpy import trapz
import os
import subprocess
import os
from scipy.stats.stats import pearsonr
from scipy.stats.stats import kendalltau
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

os.chdir('/home/nahid/Downloads/trec_eval.9.0/')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

os.chdir('/home/nahid/Downloads/trec_eval.9.0/')

base_address1 = "/home/nahid/UT_research/clueweb12/bpref_result/"
plotAddress = "/home/nahid/UT_research/clueweb12/bpref_result/plots/tau/map1/"
baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"

ranker_location = {}
ranker_location["WT2013"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/WT2013/input.ICTNET13RSR2"
ranker_location["WT2014"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/WT2014/input.Protoss"
ranker_location["gov2"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/gov2/input.indri06AdmD"
ranker_location["TREC8"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/TREC8/input.ibmg99b"



protocol_list = ['SAL','CAL', 'SPL']
#dataset_list = ['WT2013']
dataset_list = ['WT2014','WT2013', 'gov2', 'TREC8']
ranker_list = ['False']
sampling_list = ['True']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1] # skiping seed part which is named as 0.1
#train_per_centage = [0.2, 0.3] # skiping seed part which is named as 0.1
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#x_labels_set =[10,20]


result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
#subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]
subplot_loc = [221,222,223,224]
ranker_name = {}
ranker_name["WT2013"] = "Input.ICTNET13RSR2"
ranker_name["WT2014"] = "input.Protoss"
ranker_name["gov2"] = "input.indri06AdmD"
ranker_name["TREC8"] = "input.ibmg99b"

for datasource in dataset_list:  # 1
    originalqrelMap = []
    predictedqrelMap = []
    if datasource == 'gov2':
        originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
        # qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/qrels.tb06.top50.txt'
        qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modified_qreldocsgov2.txt'
        originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/'
        destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
        predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/prediction/"
        predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modifiedprediction/"
    elif datasource == 'TREC8':
        originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
        qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/relevance.txt'
        originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/'
        destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
        predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/prediction/"
        predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/modifiedprediction/"
    elif datasource == 'WT2013':
        originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
        qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modified_qreldocs2013.txt'
        originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/'
        destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
        predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/prediction/"
        predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modifiedprediction/"

    else:
        originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
        qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modified_qreldocs2014.txt'
        originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/'
        destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
        predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/prediction/"
        predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modifiedprediction/"

    #print "Original Part"

    #fileList = os.listdir(originAdress)
    #for fileName in fileList:
    system = ranker_location[datasource]
    #shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
    shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system

    #print shellCommand
    p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        #print line
        values = line.split()
        map = float(values[2])
        print datasource, "&", ranker_name[datasource], "&", str(map)[0:5], "\\\\"
        #originalqrelMap.append(map)

    retval = p.wait()




