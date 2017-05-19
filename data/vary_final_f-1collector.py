import os
from numpy import trapz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"

base_address1 = "/home/nahid/UT_research/clueweb12/vary_result_1/"
plotAddress =  "/home/nahid/UT_research/clueweb12/vary_result_1/plots/f1/"


protocol_list = ['CAL']
#dataset_list = ['WT2013','WT2014', 'gov2']
dataset_list = ['WT2013', 'WT2014', 'gov2','TREC8']
ranker_list = ['False']
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
subplot_loc = [221,222,223,224]


for use_ranker in ranker_list:
    for iter_sampling in sampling_list:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        var = 0
        s=""
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
                                for bucket in xrange(1,6):
                                    if datasource == 'TREC8' and bucket == 4:
                                        break
                                    learning_curve_location = base_address4 + str(bucket) +'/' + 'learning_curve_protocol:' + protocol + '_batch:' + str(
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
                                    list1 = list[1:len(list)]
                                    print length
                                    counter = 0
                                    protocol_result[bucket] = list1



            #plt.figure(var)
            print len(training_variation)
            plt.subplot(subplot_loc[var])
            '''plt.plot(x_labels_set, protocol_result['SAL'], '-r', label='SAL',linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['CAL'], '-b', label = 'CAL',linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['SPL'], '-g', label= 'SPL',linewidth=2.0)
'''
            '''plt.plot(x_labels_set, protocol_result['SAL'],   marker='o', color = 'black',label='SAL', linewidth=1.0)
            plt.plot(x_labels_set, protocol_result['CAL'],  marker = '^',color = 'black',label='CAL', linewidth=1.0)
            plt.plot(x_labels_set, protocol_result['SPL'],  marker = 's',color = 'black', label='SPL', linewidth=1.0)
'''

            auc_1 = trapz(protocol_result[1], dx=10)
            auc_2 = trapz(protocol_result[2], dx=10)
            auc_3 = trapz(protocol_result[3], dx=10)

            if datasource != 'TREC8':

                auc_4 = trapz(protocol_result[4], dx=10)
                auc_5 = trapz(protocol_result[5], dx=10)

                plt.plot(x_labels_set, protocol_result[1], marker='o',  label='prevalance (p) <= 10%, AUC:'+str(auc_1)[:4], linewidth=1.0)
                plt.plot(x_labels_set, protocol_result[2], marker = '^', label='11% <= p <= 20%, AUC:'+str(auc_2)[:4], linewidth=1.0)
                plt.plot(x_labels_set, protocol_result[3], marker = 's',  label='21% <= p <= 30%, AUC:'+str(auc_3)[:4], linewidth=1.0)
                plt.plot(x_labels_set, protocol_result[4], marker='v', label='31% <= p <= 40%, AUC:'+str(auc_4)[:4], linewidth=1.0)
                plt.plot(x_labels_set, protocol_result[5], marker='p', label='p > 40%, AUC:'+str(auc_5)[:4], linewidth=1.0)
            else:

                plt.plot(x_labels_set, protocol_result[1], marker='o', label='prevalance (p) <= 10%, AUC:'+str(auc_1)[:4], linewidth=1.0)
                plt.plot(x_labels_set, protocol_result[2], marker='^', label='11% <= p <= 20%, AUC:'+str(auc_2)[:4], linewidth=1.0)
                plt.plot(x_labels_set, protocol_result[3], marker='s', label='21% <= p <= 30%, AUC:'+str(auc_3)[:4], linewidth=1.0)

            plt.xlabel('Percentage of human judgements', size = 8)

            plt.ylabel('F-1 measure', size = 8)
            plt.ylim([0.5,1])
            plt.legend(loc=4, fontsize = 8)
            plt.title(datasource, size= 8)
            plt.grid()
            var = var + 1
        plt.suptitle(s1, size=8)
        plt.tight_layout()
        #plt.show()
        plt.savefig(plotAddress+s1+'.pdf', format='pdf')

    #exit(0)