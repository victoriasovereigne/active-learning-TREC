
#topicSkipList = [202,225,255, 278, 805]
topicSkipList = [202,210,225,234,235,238,244,251,255,262,269,271,278,283,289,291,803,805]
datasource = 'WT2013' # can be  dataset = ['TREC8', 'gov2', 'WT']
if datasource=='TREC8':
    processed_file_location = '/home/nahid/UT_research/TREC/TREC8/processed.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/relevance.txt'
    start_topic = 401
    end_topic = 451
elif datasource=='gov2':
    processed_file_location = '/home/nahid/UT_research/TREC/gov2/processed.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/qrels.tb06.top50.txt'
    realFile = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/realnumberofdocsgov2.txt'
    destinationAddress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modified_qreldocsgov2.txt'
    start_topic = 801
    end_topic = 851
elif datasource=='WT2013':
    processed_file_location = '/home/nahid/UT_research/clueweb12/pythonprocessed/processed_new.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/clueweb12/qrels/qrelsadhoc2013.txt'
    realFile = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/realnumberofdocs2013.txt'
    destinationAddress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modified_qreldocs2013.txt'
    start_topic = 201
    end_topic = 251
else:
    processed_file_location = '/home/nahid/UT_research/clueweb12/pythonprocessed/processed_new.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/clueweb12/qrels/qrelsadhoc2014.txt'
    realFile = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/realnumberofdocs2014.txt'
    destinationAddress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modified_qreldocs2014.txt'
    start_topic = 251
    end_topic = 301


lineCounter = 0
f = open(realFile)
print f
realqrel = []
for lines in f:
    values = lines.split()
    docNo = values[1]
    #print docNo
    realqrel.append(docNo)
    lineCounter = lineCounter + 1
f.close()

print "Real File Contains Line:", lineCounter

lineCounter = 0
modified = 0
f = open(RELEVANCE_DATA_DIR)
print f
originalqrel = []
s=''
for lines in f:
    values = lines.split(" ")
    lineCounter = lineCounter + 1
    topicNo = int(values[0])
    #print topicNo
    if topicNo in topicSkipList:
        continue
    docNo = values[2]
    if docNo not in realqrel:
        continue
    column2 = values[1]
    label = int(values[3])
    if label > 1:
        label = 1
    if label < 0:
        label = 0
    s = s+str(topicNo)+" "+str(column2)+" "+str(docNo)+" "+str(label)+"\n"
    modified = modified + 1
f.close()
print "Original Qrel File Contains Line:", lineCounter
print "Modified Qrel File Contains Line:", modified

output = open(destinationAddress, "w")
output.write(s)
output.close()