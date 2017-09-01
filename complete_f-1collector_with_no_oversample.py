import os
from scipy.integrate import simps
from numpy import trapz
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

from pprint import pprint

baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"

# base_address1 = "/home/nahid/UT_research/clueweb12/new_result/"
base_address1 = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/result/'
plotAddress =  "/v/filer4b/v20q001/vlestari/Documents/Summer/IR/result/plots/"


protocol_list = ['SAL', 'CAL', 'SPL']
#dataset_list = ['WT2013','WT2014']
# dataset_list = ['WT2014', 'WT2013', 'gov2','TREC8']
dataset_list = ['manual_keyword', 'manual_both', 'manual_summary']
ranker_list = ['False']
# sampling_list = ['True','False']
sampling_list = ['Over', 'Under']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
linestyles = ['_', '-', '--', ':']
metrics = ['F1 SCORE', 'PRECISION', 'RECALL', 'SPECIFICITY']

result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
#subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]
#subplot_loc = [411,412,413,414, 421,422,423,424, 431, 432, 433, 434, 441, 442, 443, 444]
subplot_loc = [441, 442, 443, 444, 445, 446, 447, 448, 449, 4410, 4411, 4412, 4413, 4414, 4415, 4416]


fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20,15))
var = 1
for use_ranker in ranker_list:
    for iter_sampling in sampling_list:
        s=""
        # if use_ranker == "True" and iter_sampling == "True":
        #     # base_address1 = "/home/nahid/UT_research/clueweb12/new_result/"
        #     base_address1 = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/result/' #plots/ranker/oversample/'
        # elif use_ranker == "False" and iter_sampling == "True":
        #     # base_address1 = "/home/nahid/UT_research/clueweb12/complete_result/"
        #     base_address1 = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/result/' #plots/no_ranker/oversample/'
        # elif use_ranker=="False" and iter_sampling == "False":
        #     # base_address1 = "/home/nahid/UT_research/clueweb12/nooversample_result/"
        #     base_address1 = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/result/' #plots/no_ranker/'
        # else:
        #     continue
        for datasource in dataset_list: # 1
            try:
                base_address2 = base_address1 + str(datasource) + "/"
                if use_ranker == 'True':
                    base_address3 = base_address2 + "ranker/"
                    s1="Ranker and "
                else:
                    base_address3 = base_address2 + "no_ranker/"
                    s1 = "IS and "
                if iter_sampling == 'Over':
                    base_address4 = base_address3 + "oversample/"
                    s1 = s1+"oversampling"
                elif iter_sampling == 'Under':
                    base_address4 = base_address3 + "undersample/"
                    s1 = s1+"undersampling"
                else:
                    base_address4 = base_address3 + "oversample/"
                    s1 = s1+"no oversample"

                training_variation = []
                # ==========================================================================
                # start for
                # ==========================================================================
                for seed in seed_size: # 2
                    for batch in batch_size: # 3
                        for protocol in protocol_list: #4
                                print "Dataset", datasource,"Protocol", protocol, "Seed", seed,"Batch", batch
                                s = "Dataset:"+ str(datasource)+", Seed:" + str(seed) + ", Batch:"+ str(batch)

                                for fold in xrange(1, 2):
                                    learning_curve_location = base_address4 + 'learning_curve_protocol:' + protocol + '_batch:' + str(
                                        batch) + '_seed:' + str(seed) + '_fold' + str(fold) + '.txt'

                                list = []
                                dict1 = {}

                                for metric in metrics:
                                    dict1[metric] = []

                                # reading the dataset
                                f = open(learning_curve_location)
                                # length = 0

                                whole = f.read()
                                mylist = whole.split('\n')

                                for metric in metrics:
                                    index = mylist.index(metric) + 1
                                    list1 = mylist[index]

                                    print('========================================')
                                    # print(metric)
                                    # print(list1)

                                    values = list1.split(',')
                                    length = 0

                                    if len(values) < 12:
                                        values.append('1.0')

                                    for val in values:
                                        if val == '':
                                            continue
                                        # else:
                                        dict1[metric].append(float(val))
                                        length += 1

                                    list1 = dict1[metric]

                                    if use_ranker == "True":
                                        list1 = list1[0:len(list1)-2]
                                        list1.append(list1[len(list1)-1])
                                    else:
                                        list1 = list1[1:len(list1)]

                                    print('**********************************')
                                    print(learning_curve_location, metric)
                                    print(list1)
                                    print 'length', length
                                    counter = 0

                                    protocol_result[(protocol,metric,datasource,iter_sampling,use_ranker)] = list1
                                    if protocol == 'SAL':
                                        start = 10
                                        end = start + (length - 1)*25
                                        while start <= end:
                                            training_variation.append(start)
                                            start = start + 25

                                # print(protocol_result)
                                # print('######################################')
                                # print(dict1)

                    # print len(training_variation)
                    # plt.subplot(3,4, var)

                    # auc_SAL = trapz(protocol_result[('SAL', metric)], dx=10)
                    # auc_CAL = trapz(protocol_result[('CAL', metric)], dx=10)
                    # auc_SPL = trapz(protocol_result[('SPL', metric)], dx=10)

                    # print auc_SAL, auc_CAL, auc_SPL

                    # print 'length:', len(x_labels_set), len(protocol_result[('SAL',metric)])
                    # plt.plot(x_labels_set, protocol_result[('SAL',metric)],  '-r', marker='o',  label='SAL, AUC:'+str(auc_SAL)[:4], linewidth=2.0)
                    # plt.plot(x_labels_set, protocol_result[('CAL',metric)],  '-b', marker = '^', label='CAL, AUC:'+str(auc_CAL)[:4], linewidth=2.0)
                    # plt.plot(x_labels_set, protocol_result[('SPL',metric)],  '-g', marker = 's',  label='SPL, AUC:'+str(auc_SPL)[:4], linewidth=2.0)

                    # if var > 6:
                    #     plt.xlabel('Percentage of human judgements', size = 16)

                    # if var == 1 or var == 5 or var == 9 or var == 13:
                    #     plt.ylabel(s1+'\n' + metric, size = 16)

                    # plt.ylim([0.5,1])
                    # plt.legend(loc=4, fontsize=16)
                    # plt.title(datasource, size= 16)
                    # plt.grid()
                    # var = var + 1

                    # plt.tight_layout()
                    # plt.savefig(plotAddress+s1+metric+'new_.pdf', format='pdf')
                # ==========================================================================
                # end for
                # ==========================================================================
            except Exception as e:
                # print datasource, use_ranker, iter_sampling
                print e
                continue

# for key in protocol_result.keys():
#     print(key, protocol_result[key])

for i, metric in enumerate(metrics):
    # plt.figure(i)
    var = 1
    for iter_sampling in sampling_list:
        for datasource in dataset_list:
            for use_ranker in ranker_list:
                try:
                    plt.subplot(2,3, var)
                    auc_CAL = trapz(protocol_result[('CAL', metric, datasource, iter_sampling, use_ranker)], dx=10)
                    auc_SAL = trapz(protocol_result[('SAL', metric, datasource, iter_sampling, use_ranker)], dx=10)
                    auc_SPL = trapz(protocol_result[('SPL', metric, datasource, iter_sampling, use_ranker)], dx=10)

                    print auc_SAL, auc_CAL, auc_SPL

                    print 'length:', len(x_labels_set), len(protocol_result[('SAL',metric, datasource, iter_sampling, use_ranker)])
                    plt.plot(x_labels_set, protocol_result[('CAL',metric, datasource, iter_sampling, use_ranker)],  '-b', marker = '^', label='CAL, AUC:'+str(auc_CAL)[:4], linewidth=2.0)
                    plt.plot(x_labels_set, protocol_result[('SAL',metric, datasource, iter_sampling, use_ranker)],  '-r', marker='o',  label='SAL, AUC:'+str(auc_SAL)[:4], linewidth=2.0)
                    plt.plot(x_labels_set, protocol_result[('SPL',metric, datasource, iter_sampling, use_ranker)],  '-g', marker = 's',  label='SPL, AUC:'+str(auc_SPL)[:4], linewidth=2.0)

                    # if var > 6:
                    # plt.xlabel('Percentage of human judgements', size = 16)

                    # if var == 1 or var == 5 or var == 9 or var == 13:
                    plt.ylabel(metric)

                    plt.ylim([0.1,1])
                    plt.legend(loc=4)
                    plt.title(datasource + ' ' + iter_sampling + 'sampling')
                    plt.grid()
                except Exception as e:
                    print "------------------------------------------"
                    print e
                    print metric, datasource, iter_sampling, use_ranker
                    continue

                var = var + 1

    # plt.tight_layout()
    plt.savefig(plotAddress+metric+'.pdf', format='pdf')
    plt.clf()