import os

dataset =['WT2014']
fileList = os.listdir("/media/nahid/Windows8_OS/downloadsresult/WT2014/result")
protocol_list = ['SAL', 'CAL', 'Basic']
batch_size = [25, 50, 100]
seed_size = [30, 50, 70]
#dataset = ['TREC8', 'gov2', 'WT2013', 'WT2014']

test_size_set = [0.2, 0.4, 0.6, 0.8]
result_location = ''
counter = 0
missing = 0
for datasource in dataset: # 1
    for seed in seed_size: # 2
        for batch in batch_size: # 3
            for protocol in protocol_list: #4
                    for test_size in test_size_set:
                        for fold in xrange(1,6):
                            fileName =  str(
                                test_size) + '_protocol:' + protocol + '_batch:' + str(batch) + '_seed:' + str(seed) +'_fold'+str(fold)+ '.txt'
                            counter = counter + 1
                            if fileName not in fileList:
                                print fileName
                                missing = missing + 1

print "Missing", missing
print "Total", counter