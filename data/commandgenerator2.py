import sys

protocol_list = ['SAL', 'CAL']
batch_size = [25]
seed_size = [30, 50, 70]
dataset = ['TREC8']
test_size_set = [0.2, 0.4, 0.6, 0.8]


shellcommand = '#!/bin/sh\n'
s=''
variation = 1
for datasource in dataset: # 1
    for seed in seed_size: # 2
        for batch in batch_size: # 3
            for protocol in protocol_list: #4
                for test_size in test_size_set:
                    print "python "+ "activeCode1.py " + datasource +" "+ str(seed) + " "+ str(batch) + " "+ protocol + " "+str(test_size)
                    s = s + "\npython "+ "activeCode1.py " + datasource +" "+ str(seed) + " "+ str(batch) + " "+ protocol + " "+str(test_size)
                    tmp = '#!/bin/bash\n' \
                          '#SBATCH -J activeLearning' + str(variation) + '         # job name\n' \
                                                                         '#SBATCH -o activeLearning' + str(
                        variation) + '.o%j       # output and error file name (%j expands to jobID)\n' \
                                     '#SBATCH -n 20              # total number of mpi tasks requested\n' \
                                     '#SBATCH -p gpu     # queue (partition) -- normal, development, etc.\n' \
                                     '#SBATCH -t 11:59:59        # run time (hh:mm:ss) - 1.0 hours\n' \
                                     '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                                     '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                                     '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                                     '\nmodule load python'

                    s = tmp + s
                    filname = '/home/nahid/PycharmProjects/parser/newscript2/activeJobTREC8Batch25'+ str(variation)
                    text_file = open(filname, "w")
                    text_file.write(s)
                    text_file.close()

                    s=''


                    shellcommand = shellcommand + '\nsbatch activeJobTREC8Batch25'+ str(variation)
                    variation = variation + 1


filename1 = '/home/nahid/PycharmProjects/parser/newscript2/batch_command.sh'
print shellcommand
text_file = open(filename1, "w")
text_file.write(shellcommand)
text_file.close()

print "Number of variations:" + str(variation)