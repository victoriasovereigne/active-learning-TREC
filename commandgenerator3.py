import sys


protocol_list = ['SAL', 'CAL', 'SPL']
dataset_list = ['WT2013', 'WT2014', 'gov2', 'TREC8']
ranker_list = ['True', 'False']
sampling_list = ['True', 'False']
train_per_centage_flag = 'True'

shellcommand = '#!/bin/sh\n'
s=''
variation = 0
for datasource in dataset_list: # 1
    for protocol in protocol_list: #4
        for use_ranker in ranker_list:
            for iter_sampling in sampling_list:
                if iter_sampling == 'True':
                    correction = 'False'
                else:
                    correction = 'True'
                print "python "+ "finite_pool_correction_no_ranker_percentage.py " + datasource +" "+ protocol+" "+str(use_ranker)+" "+str(iter_sampling)+" "+ str(correction)+  " "+ str(train_per_centage_flag)
                s = s + "python "+ "finite_pool_correction_no_ranker_percentage.py " + datasource +" "+ protocol+" "+str(use_ranker)+" "+str(iter_sampling)+" "+ str(correction)+  " "+ str(train_per_centage_flag)+ "\n"
                variation = variation + 1


print "number of variations:", variation
filename1 = '/home/nahid/PycharmProjects/parser/batch_command.sh'
print shellcommand
text_file = open(filename1, "w")
text_file.write(shellcommand+s)
text_file.close()

print "Number of variations:" + str(variation)