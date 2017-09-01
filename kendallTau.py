import subprocess
import os
from scipy.stats.stats import pearsonr
from scipy.stats.stats import kendalltau
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

os.chdir('/home/nahid/Downloads/trec_eval.9.0/')

'''
dataset = ['WT2013']
datasource = 'WT2013' # can be  dataset = ['TREC8', 'gov2', 'WT']
# following parameter is selected from learning curve of WT2013
# note trainsize = 0.6 so test should be 0.4
protocol_list = ['CAL']
batch_size = [25]
seed_size = [70]
test_size_set = [0.4]
'''


dataset = ['WT2013']
datasource = 'WT2013' # can be  dataset = ['TREC8', 'gov2', 'WT']
# following parameter is selected from learning curve of WT2014
# note trainsize = 0.4 so test should be 0.6
protocol_list = ['CAL']
batch_size = [25]
seed_size = [10]
test_size_set = [0.6]

'''
dataset = ['gov2']
datasource = 'gov2' # can be  dataset = ['TREC8', 'gov2', 'WT']
# following parameter is selected from learning curve of WT2014
# note trainsize = 0.4 so test should be 0.6
protocol_list = ['CAL']
batch_size = [50]
seed_size = [50]
test_size_set = [0.6]
'''

'''
dataset = ['TREC8']
datasource = 'TREC8' # can be  dataset = ['TREC8', 'gov2', 'WT']
# following parameter is selected from learning curve of WT2014
# note trainsize = 0.4 so test should be 0.6
protocol_list = ['CAL']
batch_size = [50]
seed_size = [50]
test_size_set = [0.6]
'''


if datasource=='gov2':
    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/"+datasource+"/"
    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/qrels.tb06.top50.txt'
    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/prediction/"
    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modifiedprediction/"
elif datasource=='TREC8':
    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/"+datasource+"/"
    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/relevance.txt'
    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/prediction/"
    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/modifiedprediction/"
elif datasource=='WT2013':
    originAdress = "/v/filer4b/v20q001/vlestari/Documents/Summer/IR/unzipped/"+datasource+"/"
    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modified_qreldocs2013.txt'
    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/prediction/"
    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modifiedprediction/"

else:
    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modified_qreldocs2014.txt'
    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/prediction/"
    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modifiedprediction/"


originalqrelMap = []
predictedqrelMap = []


#QrelChekcer
print "Qrel cheker part"
qrelMapper = {}
docList = []
f = open(qrelAdress)
s = ""
d = 0
for lines in f:
    values = lines.split()
    topic = values[0]
    docNo = values[2]
    label = values[3]
    if docNo in docList:
        #print "Duplicate Doc", docNo
        d = d + 1
    docList.append(docNo)
    key = topic + "##" + docNo
    #print key
    qrelMapper[key] = label
f.close()

print "Duplicate Counter:", d
predictionqrel = '/home/nahid/UT_research/clueweb12/result_ranker_over_percentage_WT2013/prediction0.2_protocol:CAL_batch:25_seed:10_fold1_oversampling:False_correction:False_iter_sampling:True_1.1.txt'

f = open(predictionqrel)
s = ""
counter = 0
for lines in f:
    values = lines.split()
    topic = values[0]
    docNo = values[1]
    label = values[2]
    key = topic + "##" + docNo
    if key in qrelMapper.keys():
        actualLabel = qrelMapper[key]
        if actualLabel != label:
            print key, actualLabel, label
            counter = counter + 1
    else:
        print key, "not found"
f.close()
print "Label Not match Counter",counter





print "Original Part"
fileList = os.listdir(originAdress)
for fileName in fileList:
    system = originAdress + fileName
    shellCommand = './trec_eval -m map '+qrelAdress+' '+ system
    print shellCommand
    p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print line
        values = line.split()
        map = float(values[2])
        originalqrelMap.append(map)

    retval = p.wait()


print "Predicted Part"
fold = 1
for datasource in dataset: # 1
    for seed in seed_size: # 2
        for batch in batch_size: # 3
            for protocol in protocol_list: #4
                for test_size in test_size_set:
                    #print "python "+ "activeCode.py " + datasource +" "+ str(seed) + " "+ str(batch) + " "+ protocol, str(variation)
                    #predictionqrel = predictionAddress + str(test_size) + '_protocol:' + protocol + '_batch:' + str(batch) + '_seed:' + str(seed) +'_fold'+str(fold)+ '.txt'

                    predictionqrel =  '/home/nahid/UT_research/clueweb12/result_ranker_over_percentage_WT2013/prediction0.2_protocol:CAL_batch:25_seed:10_fold1_oversampling:False_correction:False_iter_sampling:True_0.1.txt'

                    f = open(predictionqrel)
                    s=""
                    for lines in f:
                        values = lines.split()
                        topic = values[0]
                        docNo = values[1]
                        label = values[2]
                        s=s+str(topic)+" "+str(0)+" "+str(docNo)+" "+str(label)+"\n"
                    f.close()

                    #predictionModifiedqrel =  predictionModifiedAddress +str(test_size) + '_protocol:' + protocol + '_batch:' + str(batch) + '_seed:' + str(seed) +'_fold'+str(fold)+ '.txt'
                    predictionModifiedqrel = '/home/nahid/UT_research/clueweb12/result_ranker_over_percentage_WT2013/prediction0.2_protocol:CAL_batch:25_seed:10_fold1_oversampling:False_correction:False_iter_sampling:True_0.1_modified.txt'

                    output = open(predictionModifiedqrel, "w")
                    output.write(s)
                    output.close()

                    fileList = os.listdir(originAdress)
                    for fileName in fileList:
                        system = originAdress + fileName
                        shellCommand = './trec_eval -m map ' + predictionModifiedqrel + ' ' + system
                        print shellCommand
                        p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        for line in p.stdout.readlines():
                            print line
                            values = line.split()
                            map = float(values[2])
                            predictedqrelMap.append(map)

                        retval = p.wait()



fig = plt.figure(figsize=(17,5))
percentile_list = pd.DataFrame(
    {'original': originalqrelMap,
     'predicted': predictedqrelMap
    })
seaborn.regplot(x='original', y='predicted', fit_reg=True, data=percentile_list)
fig.tight_layout()
plt.show()


tau, p_value = kendalltau(originalqrelMap,predictedqrelMap)
print "Kendall's Tau", tau, "and p_value:", p_value

tau, p_value = pearsonr(originalqrelMap,predictedqrelMap)
print "Pearson's r", tau, "and p_value:", p_value