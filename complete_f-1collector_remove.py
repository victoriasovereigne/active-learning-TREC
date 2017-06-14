import os
from scipy.integrate import simps
from numpy import trapz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"

base_address1 = "/home/nahid/UT_research/clueweb12/new_result/"
plotAddress =  "/home/nahid/UT_research/clueweb12/new_result/plots/f1new/"


protocol_list = ['SAL', 'CAL', 'SPL']
#dataset_list = ['WT2013','WT2014']
dataset_list = ['WT2014', 'WT2013', 'gov2','TREC8']
ranker_list = ['True', 'False']
sampling_list = ['True']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
linestyles = ['_', '-', '--', ':']


result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
#subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]
#subplot_loc = [411,412,413,414, 421,422,423,424, 431, 432, 433, 434, 441, 442, 443, 444]
subplot_loc = [441, 442, 443, 444, 445, 446, 447, 448, 449, 4410, 4411, 4412, 4413, 4414, 4415, 4416]


fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
var = 1
for use_ranker in ranker_list:
    for iter_sampling in sampling_list:
        s=""
        if use_ranker == "True":
            base_address1 = "/home/nahid/UT_research/clueweb12/new_result/"
        else:
            base_address1 = "/home/nahid/UT_research/clueweb12/complete_result/"

        for datasource in dataset_list: # 1
            base_address2 = base_address1 + str(datasource) + "/"
            if use_ranker == 'True':
                base_address3 = base_address2 + "ranker/"
                s1="Ranker and "
            else:
                base_address3 = base_address2 + "no_ranker/"
                s1 = "Interactive Search and "
            if iter_sampling == 'True':
                base_address4 = base_address3 + "oversample/"
                s1 = s1+"oversampling"
            else:
                base_address4 = base_address3 + "htcorrection/"
                s1 = s1+"HT correction"

            training_variation = []
            for seed in seed_size: # 2
                for batch in batch_size: # 3
                    for protocol in protocol_list: #4
                            print "Dataset", datasource,"Protocol", protocol, "Seed", seed,"Batch", batch
                            s = "Dataset:"+ str(datasource)+", Seed:" + str(seed) + ", Batch:"+ str(batch)

                            for fold in xrange(1, 2):
                                learning_curve_location = base_address4 + 'learning_curve_protocol:' + protocol + '_batch:' + str(
                                    batch) + '_seed:' + str(seed) + '_fold' + str(fold) + '.txt'

                            list = []

                            f = open(learning_curve_location)
                            length = 0
                            for lines in f:
                                values = lines.split(",")
                                for val in values:
                                    if val == '':
                                        continue
                                    list.append(float(val))
                                    length = length + 1
                                break
                            print list
                            #list1 = list[1:len(list)]
                            if use_ranker == "True":
                                list1 = list[0:len(list)-2]
                                list1.append(list[len(list)-1])
                            else:
                                list1 = list[1:len(list)]
                            print length
                            counter = 0
                            protocol_result[protocol] = list1
                            if protocol == 'SAL':
                                start = 10
                                end = start + (length - 1)*25
                                while start <= end:
                                    training_variation.append(start)
                                    start = start + 25


            #plt.figure(var)
            print len(training_variation)
            #plt.subplot(subplot_loc[var])
            plt.subplot(2,4, var)
            '''plt.plot(x_labels_set, protocol_result['SAL'], '-r', label='SAL',linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['CAL'], '-b', label = 'CAL',linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['SPL'], '-g', label= 'SPL',linewidth=2.0)
'''
            '''plt.plot(x_labels_set, protocol_result['SAL'],   marker='o', color = 'black',label='SAL', linewidth=1.0)
            plt.plot(x_labels_set, protocol_result['CAL'],  marker = '^',color = 'black',label='CAL', linewidth=1.0)
            plt.plot(x_labels_set, protocol_result['SPL'],  marker = 's',color = 'black', label='SPL', linewidth=1.0)
'''

            auc_SAL = trapz(protocol_result['SAL'], dx=10)
            auc_CAL = trapz(protocol_result['CAL'], dx=10)
            auc_SPL = trapz(protocol_result['SPL'], dx=10)

            print auc_SAL, auc_CAL, auc_SPL
            #exit(0)


            plt.plot(x_labels_set, protocol_result['SAL'],  '-r', marker='o',  label='SAL, AUC:'+str(auc_SAL)[:4], linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['CAL'],  '-b', marker = '^', label='CAL, AUC:'+str(auc_CAL)[:4], linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['SPL'],  '-g', marker = 's',  label='SPL, AUC:'+str(auc_SPL)[:4], linewidth=2.0)

            if var > 12:
                plt.xlabel('Percentage of human judgements', size = 16)

            if var == 1 or var == 5 or var == 9 or var == 13:
                plt.ylabel(s1+'\n F1 measure', size = 16)
                #plt.yticks(True)
            plt.ylim([0.5,1])
            #plt.tick_params(axis='x',          # changes apply to the x-axis
            #which='both',      # both major and minor ticks are affected
            #bottom='off',      # ticks along the bottom edge are off
            #top='off',         # ticks along the top edge are off
            #labelbottom='off') # labels along the bottom edge are off)
            #if var == 1:
            #if var == 7 or var == 8:
            #    plt.legend(loc=2, fontsize = 16)
            #else:
            plt.legend(loc=4, fontsize=16)
            plt.title(datasource, size= 16)
            plt.grid()
            var = var + 1

#plt.suptitle(s1, size=10)
plt.tight_layout()

#plt.show()
plt.savefig(plotAddress+s1+'new_.pdf', format='pdf')

    #exit(0)