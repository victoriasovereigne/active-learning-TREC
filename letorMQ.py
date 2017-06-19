'''Active learning for labeling the relevant document for TREC-8 dataset
@author: Md Mustafizur Rahman (nahid@utexas.edu)'''

import os
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
import nltk
import copy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.svm import LinearSVC
import sklearn.svm.libsvm

import pickle

import logging
logging.basicConfig()

np.random.seed(1335)
TEXT_DATA_DIR = '/home/nahid/TREC/data/'
RELEVANCE_DATA_DIR = '/home/nahid/TREC/relevance.txt'
#FEATURE_DATA_DIR = '/home/nahid/OHSUMED/Feature-min/'
fileName = {'trainingset.txt', 'validationset.txt', 'testset.txt'}

FEATURE_DIR_LIST = []
FEATURE_DIR_LIST.append('/media/nahid/Windows8_OS/Gov/MQ/MQ2007/')
FEATURE_DIR_LIST.append('/media/nahid/Windows8_OS/Gov/MQ/MQ2008/')

# key-> fold number value -> list of document

train_fold = {}
validation_fold = {}
test_fold = {}

topic_number = '410'
docrepresentation = "TF-IDF"  # can be BOW, TF-IDF
sampling=False # can be True or False
test_size = 0.6    # the percentage of samples in the dataset that will be
n_labeled = 10      # number of samples that are initially labeled
preloaded = False




def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


def run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size):
    E_in, E_out = [], []

    for _ in range(quota):


        # Standard usage of libact objects
        ask_id = qs.make_query()
        #print  ask_id
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)


        model.train(trn_ds)
        #model.predict(tst_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out, model




topic_to_doclist = {} # key is the topic(string) and value is the list of docNumber
docNo_label = {} # key is the DocNo and the value is the label
all_reviews = {}
print('Reading the fold information')

X_train = []
X_test = []
y_train = []
y_test = []

if preloaded==False:

    year = 2007
    fold = 1
    for dir in FEATURE_DIR_LIST:
        #foldNumber = 1
        fold = 1
        foldNumber = str(fold) + str(year)
        for name in sorted(os.listdir(dir)):
            print name
            # if name == "ft":
            path = os.path.join(dir, name)
            print path
            for fname in sorted(os.listdir(path)):
                print fname
                # if name == "ft":
                fpath = os.path.join(path, fname)
                print fpath
                # file open
                f = open(fpath)
                print f
                tmplist = []
                docList = []
                for lines in f:
                    values = lines.split()

                    label = 1 if int(values[0])>=1 else 0
                    #label =  label >= 1? 1: 0;
                    topic = values[1][4:] # qid:
                    #print "topic", topic
                    # appending topic id with docid becuase for differnet topic id we can have same document
                    # but appending topic will make that different
                    docNo = values[len(values)-7].replace(" ", "").replace("\t", "")+"#"+topic
                    #print docNo
                    docNo_label[docNo] = label
                    docList.append(docNo)
                    features = []
                    for i in xrange(2,len(values)-6): # substracting three for three columns
                        #print values[i]
                        if "#" in values[i]:
                            break
                        index = values[i].index(":") + 1
                        #print "index", index
                        #print float(values[i][index:])
                        #print "i",i
                        features.append(float(values[i][index:]))

                    #print 'feature size',len(features)
                    all_reviews[docNo] = features

                    if(topic_to_doclist.has_key(topic)):
                        tmplist.append(docNo)
                    else:
                        tmplist = []
                        tmplist.append(docNo)
                        topic_to_doclist[topic] = tmplist

                f.close()
                if fname == "test.txt":
                    if test_fold.has_key(foldNumber):
                        tmpDocList = test_fold[foldNumber] + docList;
                        test_fold[foldNumber] = tmpDocList
                    else:
                        test_fold[foldNumber] = docList
                elif fname == "train.txt":
                    if train_fold.has_key(foldNumber):
                        tmpDocList = train_fold[foldNumber] + docList;
                        train_fold[foldNumber] = tmpDocList
                    else:
                        train_fold[foldNumber] = docList
                else:
                    if validation_fold.has_key(foldNumber):
                        tmpDocList = validation_fold[foldNumber] + docList;
                        validation_fold[foldNumber] = tmpDocList
                    else:
                        validation_fold[foldNumber] = docList
            fold = fold + 1
            foldNumber = str(fold) + str(year)
            print foldNumber, year
        print "===================================================="
        year = 2008

    print len(train_fold)
    for year in xrange(2007, 2009):
        for i in xrange(1,6):

            key = str(i)+str(year)
            print "year:", year ," fold->", i
            docList = train_fold[key]
            testList = test_fold[key]
            valiList = validation_fold[key]
            '''for doc in docList:
                print doc
            '''
            print len(docList), len(valiList), len(testList)
            print "\n"


    print "total doc", len(all_reviews)

    s = "";
    #for topic in sorted(topic_to_doclist.keys()):
    leave = 5


    #for fold in xrange(1,2):

    '''
    fold = 1
    docList = train_fold[fold]
    for documentNo in docList:
        X_train.append(all_reviews[documentNo])
        y_train.append(docNo_label[documentNo])
    docList = validation_fold[fold]
    for documentNo in docList:
        X_train.append(all_reviews[documentNo])
        y_train.append(docNo_label[documentNo])

    docList = test_fold[fold]
    for documentNo in docList:
        X_test.append(all_reviews[documentNo])
        y_test.append(docNo_label[documentNo])

        #leave = leave - 1
    '''

    fold = 5
    for year in xrange(2007, 2009):
        key = str(fold) + str(year)
        if year==2008:
            docList = train_fold[key]
            for documentNo in docList:
                X_train.append(all_reviews[documentNo])
                y_train.append(docNo_label[documentNo])
            docList = validation_fold[key]
            for documentNo in docList:
                X_train.append(all_reviews[documentNo])
                y_train.append(docNo_label[documentNo])

            docList = test_fold[key]
            for documentNo in docList:
                X_test.append(all_reviews[documentNo])
                y_test.append(docNo_label[documentNo])
        if year==2007:
            docList = train_fold[key]
            for documentNo in docList:
                X_train.append(all_reviews[documentNo])
                y_train.append(docNo_label[documentNo])
            docList = validation_fold[key]
            for documentNo in docList:
                X_train.append(all_reviews[documentNo])
                y_train.append(docNo_label[documentNo])

            docList = test_fold[key]
            for documentNo in docList:
                X_train.append(all_reviews[documentNo])
                y_train.append(docNo_label[documentNo])

    output = open('/home/nahid/TREC/data1/X_train.txt', 'ab+')
    pickle.dump(X_train, output)
    output.close()

    output = open('/home/nahid/TREC/data1/y_train.txt', 'ab+')
    pickle.dump(y_train, output)
    output.close()

    output = open('/home/nahid/TREC/data1/X_test.txt', 'ab+')
    pickle.dump(X_test, output)
    output.close()

    output = open('/home/nahid/TREC/data1/y_test.txt', 'ab+')
    pickle.dump(y_test, output)
    output.close()
else:
    input = open('/home/nahid/TREC/data1/X_train.txt', 'rb')
    X_train = pickle.load(input)
    input.close()

    input = open('/home/nahid/TREC/data1/y_train.txt', 'rb')
    y_train = pickle.load(input)
    input.close()

    input = open('/home/nahid/TREC/data1/X_test.txt', 'rb')
    X_test = pickle.load(input)
    input.close()

    input = open('/home/nahid/TREC/data1/y_test.txt', 'rb')
    y_test = pickle.load(input)
    input.close()

    print "pickle loaded"




datasize = len(X_train)
print "Whole Dataset size: ", datasize

#print len(y)
#print y
numberOne = y_train.count(1)
print "Number of One", numberOne

numberZero = y_train.count(0)
print "Number of zero", numberZero



X_train= np.array(X_train)
X_test = np.array(X_test)

#X_train = X_train[1:500]
#X_test = X_test[1:100]

y_train= np.array(y_train)
y_test = np.array(y_test)

#y_train= y_train[1:500]
#y_test = y_test[1:100]


print X_train.shape
print X_test.shape


n_labeled = int(len(y_train)*0.70)

trn_ds = Dataset(X_train, np.concatenate(
    [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
tst_ds = Dataset(X_test, y_test)
fully_labeled_trn_ds = Dataset(X_train, y_train)

trn_ds2 = copy.deepcopy(trn_ds)
lbr = IdealLabeler(fully_labeled_trn_ds)

quota = len(y_train) - n_labeled    # number of samples to query
print "quotas:", quota

#quota = 500

batch_size = int(quota / 10)
quota = 1

print X_train[1]
print type(X_train[1][1])

#fast_SVM = LinearSVC(random_state=101, penalty='l2', loss='l1')
#fast_SVM = svm.NuSVC()
'''fast_SVM = clf = svm.SVC(probability=False,  # cache_size=200,
              kernel="rbf", C=2.8, gamma=.0073)

fast_SVM.fit(X_train, y_train)
y_pred = fast_SVM.predict(X_test)
'''

'''model = svm.SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
'''


# Comparing UncertaintySampling strategy with RandomSampling.
# model is the base learner, e.g. LogisticRegression, SVM ... etc.
qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
model = LogisticRegression()


E_in_1, E_out_1, model = run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size)
y_pred = model.predict(X_test)

num_correct = (y_pred == y_test).sum()
print "Number of correct:", num_correct
recall = num_correct / len(y_test)
print "model accuracy (%): ", recall * 100, "%"


precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1score = f1_score(y_test, y_pred, average='binary')

print "precision score:", precision
print "recall score:", recall
print "f-1 score:", f1score

num_correct = (y_pred == y_test).sum()
recall = num_correct / len(y_test)
print "model accuracy (%): ", recall * 100, "%"


s=""
s=s+topic+","+str(datasize)+","+str(numberOne)+","+str(numberZero)+","+str(precision)+","+str(recall)+","+str(f1score)+"\n";

text_file = open("/home/nahid/TREC/results_analysis_450_SVM_MQ.txt", "w")
text_file.write(s)
text_file.close()

# Plot the learning curve of UncertaintySampling to RandomSampling
# The x-axis is the number of queries, and the y-axis is the corresponding
# error rate.
query_num = np.arange(1, quota + 1)
#plt.plot(query_num, E_in_1, 'g', label='qs Ein')
#plt.plot(query_num, E_in_2, 'r', label='random Ein')
plt.plot(query_num, E_in_1, 'b', label='')
print E_out_1
plt.xlabel('Number of Queries')
plt.ylabel('Error')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
#plt.show()
exit(0)

