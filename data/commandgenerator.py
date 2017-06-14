import sys

protocol_list = ['SAL', 'CAL', 'Basic']
batch_size = [25, 50, 100]
seed_size = [30, 50, 70]
dataset = ['TREC8', 'gov2', 'WT2013', 'WT2014']

shellcommand = '#!/bin/sh\n'

variation = 1
for datasource in dataset: # 1
    for seed in seed_size: # 2
        for batch in batch_size: # 3
            for protocol in protocol_list: #4
                print "python "+ "activeCode.py " + datasource +" "+ str(seed) + " "+ str(batch) + " "+ protocol, str(variation)
                s = '#!/bin/bash\n' \
                    '#SBATCH -J activeLearning'+str(variation) +'         # job name\n' \
                    '#SBATCH -o activeLearning'+str(variation) +'.o%j       # output and error file name (%j expands to jobID)\n' \
                    '#SBATCH -n 1              # total number of mpi tasks requested\n' \
                    '#SBATCH -p gpu     # queue (partition) -- normal, development, etc.\n' \
                    '#SBATCH -t 10:59:59        # run time (hh:mm:ss) - 1.0 hours\n' \
                    '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                    '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                    '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                    '\nmodule load python'
                s = s + "\n\npython "+ "activeCode.py " + datasource +" "+ str(seed) + " "+ str(batch) + " "+ protocol

                filname = '/home/nahid/PycharmProjects/parser/script/activeJob'+ str(variation)
                text_file = open(filname, "w")
                text_file.write(s)
                text_file.close()

                shellcommand = shellcommand + '\nsbatch activeJob'+ str(variation)
                variation = variation + 1


filename1 = '/home/nahid/PycharmProjects/parser/script/batch_command.sh'
print shellcommand
text_file = open(filename1, "w")
text_file.write(shellcommand)
text_file.close()


print "Number of variations:" + str(variation)