from scipy.stats.stats import kendalltau
from numpy import trapz
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

os.chdir('/home/nahid/Downloads/trec_eval.9.0/')

base_address1 = "/home/nahid/UT_research/clueweb12/bpref_result/"
plotAddress = "/home/nahid/UT_research/clueweb12/bpref_result/plots/tau/map1/"
baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"

import matplotlib.pyplot as plt
import numpy as np

plotAddress = '/home/nahid/UT_research/clueweb12/bpref_result/plots/'

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
map_score = {}


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

    print "Original Part"

    originalMapResult = originalMapResult + 'map.txt'
    f = open(originalMapResult)
    length = 0
    tmplist = []
    for lines in f:
        values = lines.split(",")
        for val in values:
            if val == '':
                continue
            tmplist.append(float(val))
            length = length + 1
        break
    originalqrelMap = tmplist
    print originalqrelMap
    print len(originalqrelMap)

    map_score[datasource] = originalqrelMap
    originalqrelMap = []


data = [map_score['WT2014'], map_score['WT2013'],map_score['gov2'], map_score['TREC8']]
plt.boxplot( data)
plt.xticks([1, 2, 3, 4], ['WT2014', 'WT2013', 'gov2', 'TREC8'])
plt.grid()
plt.xlabel("Data set",size = 16)
plt.ylabel("MAP score for submitted ranking \n systems using original qrel", size = 16)
#plt.ylim(0,100)
#plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
#plt.show()


print "WT2014 & input.Protoss & 0.181 & ", str(np.mean(map_score['WT2014']))[0:5], "&",str(np.std(map_score['WT2014']))[0:5], "\\\\"

print "WT2013 & Input.ICTNET13RSR2 & 0.111 &" ,str(np.mean(map_score['WT2013']))[0:5], "&",str(np.std(map_score['WT2013']))[0:5], "\\\\"
print "gov2 & input.indri06AdmD & 0.35 &", str(np.mean(map_score['gov2']))[0:5], "&",str(np.std(map_score['gov2']))[0:5], "\\\\"
print "TREC8 & input.ibmg99b & 0.260 &", str(np.mean(map_score['TREC8']))[0:5], "&",str(np.std(map_score['TREC8']))[0:5], "\\\\"
plt.tight_layout()
# plt.show()
plt.savefig(plotAddress +'mapBoxPlot.pdf', format='pdf')

exit(0)
