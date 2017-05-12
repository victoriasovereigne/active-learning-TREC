import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"
plotAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/plots/"
#dataset =['TREC8', 'gov2','WT2013', 'WT2014']
dataset = ['WT2013', 'WT2014', 'gov2','TREC8']
protocol_list = ['SAL', 'CAL', 'SPL']
n_labeled =  10 #50      # number of samples that are initially labeled
batch_size = 25 #50
sampling=False # can be True or False
iter_sampling=True
correction = False
if correction == True:
    sampling = False  # can be True or False
    iter_sampling = False

batch_size = [25]
seed_size = [10]
training_variation = []
test_size_set = [0.2]
result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
#subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]
subplot_loc = [221,222,223,224]

var = 0
plt.subplots(nrows=2, ncols=2)
for datasource in dataset: # 1

    training_variation = []

    for seed in seed_size: # 2
        for batch in batch_size: # 3
            for protocol in protocol_list: #4
                    print "Dataset", datasource,"Protocol", protocol, "Seed", seed,"Batch", batch
                    s = "Dataset:"+ str(datasource)+", Seed:" + str(seed) + ", Batch:"+ str(batch)

                    for test_size in test_size_set:

                        for fold in xrange(1, 2):
                            result_location = '/home/nahid/UT_research/clueweb12/result_over_' + str(
                                datasource) + '/' + str(
                                test_size) + '_protocol:' + protocol + '_batch:' + str(batch) + '_seed:' + str(
                                seed) + '_fold' + str(fold) + '_oversampling:' + str(
                                sampling) + '_correction:' + str(correction) + '_iter_sampling:' + str(
                                iter_sampling) + '.txt'
                            predicted_location = '/home/nahid/UT_research/clueweb12/result_over_' + str(
                                datasource) + '/prediction' + str(
                                test_size) + '_protocol:' + protocol + '_batch:' + str(batch) + '_seed:' + str(
                                seed) + '_fold' + str(fold) + '_oversampling:' + str(
                                sampling) + '_correction:' + str(correction) + '_iter_sampling:' + str(
                                iter_sampling) + '.txt'

                            learning_curve_location = '/home/nahid/UT_research/clueweb12/result_over_' + str(
                                datasource) + '/learning_curve' + str(
                                test_size) + '_protocol:' + protocol + '_batch:' + str(batch) + '_seed:' + str(
                                seed) + '_fold' + str(fold) + '_oversampling:' + str(
                                sampling) + '_correction:' + str(correction) + '_iter_sampling:' + str(
                                iter_sampling) + '.txt'

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
                    print length
                    counter = 0
                    protocol_result[protocol] = list
                    if protocol == 'SAL':
                        start = 10
                        end = start + (length - 1)*25
                        while start <= end:
                            training_variation.append(start)
                            start = start + 25



    #plt.figure(var)
    print len(training_variation)
    plt.subplot(subplot_loc[var])
    plt.plot(training_variation, protocol_result['SAL'], '-r', label='CAL',linewidth=2.0)
    plt.plot(training_variation, protocol_result['CAL'], '-b', label = 'SAL',linewidth=2.0)
    plt.plot(training_variation, protocol_result['SPL'], '-g', label= 'SPL',linewidth=2.0)

    plt.xlabel('Number of human judgements')

    plt.ylabel('F-1 measure')
    plt.ylim([0.2,1])
    plt.legend(loc=2)
    plt.title(s)
    plt.grid()
    var = var + 1

#plt.suptitle("Dataset:"+ datasource,size=16)

plt.tight_layout()
plt.show()
    #plt.savefig(plotAddress+datasource+'.pdf', format='pdf')
    #exit(0)