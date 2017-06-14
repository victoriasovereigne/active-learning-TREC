import os

address = "/media/nahid/Windows8_OS/downloads/WT2014/prediction/"
dataset =['WT2014']
fileList = os.listdir(address)

qrel = '/home/nahid/UT_research/clueweb12/qrels/qrelsadhoc2014.txt'
file1 = address + '0.2_protocol:Basic_batch:50_seed:70_fold1.txt'
file2 = address + '0.2_protocol:Basic_batch:25_seed:50_fold1.txt'

docNo_label = {}
f = open(qrel)
for lines in f:
    values = lines.split()
    docNo = values[2]
    label = int(values[3])
    if label > 1:
        label = 1
    if label < 0:
        label = 0
    docNo_label[docNo] = label
f.close()

f1 = open(file1)
f2 = open(file2)

fileName = {}
matched = 0
notmatched = 0
linecounter1 = 0
for lines in f1:
    values = lines.split()
    fileName[values[0]] = int(values[1])
    linecounter1 = linecounter1 + 1
    if docNo_label.has_key(values[0]):
        if int(values[1]) == docNo_label[values[0]]:
            matched = matched + 1
        else:
            notmatched = notmatched + 1


print "Qrel Matched:", matched
print "Qrel Not Matched", notmatched

acc = (1.0)*matched / (matched + notmatched)
print "Accuracy:", acc*100

matched = 0
notmatched = 0
linecounter2 = 0
for lines in f2:
    values = lines.split()
    if fileName.has_key(values[0]):
        if int(values[1]) == fileName[values[0]]:
           matched = matched + 1
        else:
            notmatched = notmatched + 1
    linecounter2 = linecounter2 + 1



print "Matched:", matched
print "Not Matched", notmatched

if linecounter1 == linecounter2:
    print "Same", linecounter1
else:
    print "Not Same"