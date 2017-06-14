import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"
plotAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/plots/"
dataset =['TREC8', 'gov2','WT2013', 'WT2014']
protocol_list = ['SAL', 'CAL', 'Basic']
batch_size = [25, 50, 100]
seed_size = [30, 50, 70]
training_variation = [0.2, 0.4, 0.6, 0.8]
test_size_set = [0.8, 0.6, 0.4, 0.2]
result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]


for datasource in dataset: # 1
    var = 1
    plt.subplots(nrows=3, ncols=3)
    for seed in seed_size: # 2
        for batch in batch_size: # 3
            for protocol in protocol_list: #4
                    print "Dataset", datasource,"Protocol", protocol, "Seed", seed,"Batch", batch
                    s = "Seed:" + str(seed) + ", Batch:"+ str(batch)
                    list = []
                    for test_size in test_size_set:
                        cross_f1 = 0
                        for fold in xrange(1,6):
                            fileName =  str(
                                test_size) + '_protocol:' + protocol + '_batch:' + str(batch) + '_seed:' + str(seed) +'_fold'+str(fold)+ '.txt'
                            fileAddress = baseAddress + datasource +"/result/" + fileName
                            f = open(fileAddress)
                            f1 = 0
                            counter = 0
                            for lines in f:
                                values = lines.split(",")
                                f1 = f1+float(values[7])
                                counter = counter + 1
                            avg_f1 = f1/counter
                            cross_f1 = cross_f1 + avg_f1

                        avg_cross_f1 = cross_f1/5.0
                        list.append(avg_cross_f1)
                        #list.reverse()
                        #print "TestSize:", test_size, avg_cross_f1
                    print list
                    protocol_result[protocol] = list
            #plt.figure(var)

            plt.subplot(subplot_loc[var-1])
            plt.plot(training_variation, protocol_result['SAL'], '-r', label='CAL')
            plt.plot(training_variation, protocol_result['CAL'], '-b', label = 'SAL')
            plt.plot(training_variation, protocol_result['Basic'], '-g', label= 'Basic',linewidth=2.0)

            if(var==9 or var==7 or var==8):
                plt.xlabel('Training Set Size')

            plt.ylabel('F-1 measure')
            plt.title(s)
            plt.grid()
            var = var + 1
    plt.suptitle("Dataset:"+ datasource,size=16)
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()
    #plt.savefig(plotAddress+datasource+'.pdf', format='pdf')
    #exit(0)