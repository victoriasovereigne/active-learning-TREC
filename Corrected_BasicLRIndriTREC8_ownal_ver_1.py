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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import collections
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
import pickle
from math import log
import pandas as pd
import Queue

import logging
logging.basicConfig()

TEXT_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/IndriData/'
RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/relevance.txt'
docrepresentation = "TF-IDF"  # can be BOW, TF-IDF
sampling=True # can be True or False
test_size = 0    # the percentage of samples in the dataset that will be
#test_size_set = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
test_size_set = [0.2]
ranker_location = {}
ranker_location["WT2013"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/WT2013/input.ICTNET13RSR2"

datasource = 'WT2013' # can be  dataset = ['TREC8', 'gov2', 'WT']
n_labeled =  20 #50      # number of samples that are initially labeled
batch_size = 25 #50
protocol = 'CAL' #'SAL' can be ['SAL', 'CAL', 'SPL']
preloaded = True

processed_file_location = ''
start_topic = 0
end_topic = 0

if datasource=='TREC8':
    processed_file_location = '/home/nahid/UT_research/TREC/TREC8/processed.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/relevance.txt'
    start_topic = 401
    end_topic = 402
elif datasource=='gov2':
    processed_file_location = '/home/nahid/UT_research/TREC/gov2/processed.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/qrels.tb06.top50.txt'
    start_topic = 801
    end_topic = 802
elif datasource=='WT2013':
    processed_file_location = '/home/nahid/UT_research/clueweb12/pythonprocessed/processed_new.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/clueweb12/qrels/qrelsadhoc2013.txt'
    start_topic = 201
    end_topic = 251
else:
    processed_file_location = '/home/nahid/UT_research/clueweb12/pythonprocessed/processed_new.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/clueweb12/qrels/qrelsadhoc2014.txt'
    start_topic = 251
    end_topic = 301


#print result_location
#exit(0)
class relevance(object):
    def __init__(self, priority, index):
        self.priority = priority
        self.index = index
        return
    def __cmp__(self, other):
        return -cmp(self.priority, other.priority)

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




all_reviews = {}

if preloaded==False:
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        #print name
        #if name == "ft":
        path = os.path.join(TEXT_DATA_DIR, name)
        print path

        f = open(path)



        docNo = name[0:name.index('.')]
        #print docNo

        # counting the line number until '---Terms---'
        count = 0
        for lines in f:
            if lines.find("Terms")>0:
                count = count + 1
                break
            count = count + 1

        # skipping the lines until  '---Terms---' and reading the rest
        c = 0
        tmpStr = ""
        #print "count:", count
        #f = open(path)
        for lines in f:
            if c < count:
                c = c + 1
                continue
            values = lines.split()
            c = c + 1
            #print values[0], values[1], values[2]
            tmpStr = tmpStr + " "+ str(values[2])
        print tmpStr
        #exit(0)

        #if docNo in docNo_label:
        all_reviews[docNo] = (review_to_words(tmpStr))

        f.close()

    output = open(processed_file_location, 'ab+')
    # data = {'a': [1, 2, 3], }

    pickle.dump(all_reviews, output)
    output.close()

else:
    input = open(processed_file_location, 'rb')
    all_reviews = pickle.load(input)
    print "pickle loaded"


print('Reading the Ranker label Information')
f = open(ranker_location[datasource])
print "Ranker:", f
tmplist = []
Ranker_topic_to_doclist = {}
for lines in f:
    values = lines.split()
    topicNo = values[0]
    docNo = values[2]
    if (Ranker_topic_to_doclist.has_key(topicNo)):
        tmplist.append(docNo)
        Ranker_topic_to_doclist[topicNo] = tmplist
    else:
        tmplist = []
        tmplist.append(docNo)
        Ranker_topic_to_doclist[topicNo] = tmplist
f.close()
# print len(topic_to_doclist)

for test_size in test_size_set:
    seed = 1335
    for fold in xrange(1,2):
        np.random.seed(seed)
        seed = seed + fold
        result_location = '/home/nahid/UT_research/clueweb12/result3/' + str(
            test_size) + '_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+  '_oversampling:'+str(sampling)+ '.txt'
        predicted_location = '/home/nahid/UT_research/clueweb12/result3/prediction' + str(
            test_size) + '_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_oversampling:'+str(sampling)+ '.txt'

        s = "";
        pred_str = ""
        #for topic in sorted(topic_to_doclist.keys()):
        for topic in xrange(start_topic,end_topic):
            print "Topic:", topic
            if topic == 202 or topic == 212 or topic ==225 or topic == 239:
                print "Skipping Topic 202"
                continue
            topic = str(topic)

            topic_to_doclist = {}  # key is the topic(string) and value is the list of docNumber
            docNo_label = {}  # key is the DocNo and the value is the label
            docIndex_DocNo = {} # key is the index used in my code value is the actual DocNo
            docNo_docIndex = {} # key is the DocNo and the value is the index assigned by my code
            best_f1 = 0.0  # best f1 considering per iteraton of active learning
            print('Reading the relevance label')
            # file open
            f = open(RELEVANCE_DATA_DIR)
            print f
            tmplist = []
            #g = 0
            for lines in f:
                #print lines
                #g = g + 1
                #if g>2739:
                #    break
                values = lines.split()
                topicNo = values[0]
                docNo = values[2]
                label = int(values[3])
                if label > 1:
                    label = 1
                if label < 0:
                    label = 0
                docNo_label[docNo] = label
                if (topic_to_doclist.has_key(topicNo)):
                    tmplist.append(docNo)
                    topic_to_doclist[topicNo] = tmplist
                else:
                    tmplist = []
                    tmplist.append(docNo)
                    topic_to_doclist[topicNo] = tmplist
            f.close()
            #print len(topic_to_doclist)
            docList = topic_to_doclist[topic]
            print 'number of documents', len(docList)
            #print docList
            #print ('Processing news text for topic number')
            relevance_label = []
            judged_review = []

            docIndex = 0
            for documentNo in docList:
                if all_reviews.has_key(documentNo):
                    #print "in List", documentNo
                    #print documentNo, 'len:', type(all_reviews[documentNo])

                    #print all_reviews[documentNo]
                    #exit(0)
                    docIndex_DocNo[docIndex] = documentNo
                    docNo_docIndex[documentNo] = docIndex
                    docIndex = docIndex + 1
                    judged_review.append(all_reviews[documentNo])
                    relevance_label.append(docNo_label[documentNo])


            if docrepresentation == "TF-IDF":
                print "Using TF-IDF"
                vectorizer = TfidfVectorizer( analyzer = "word",   \
                                         tokenizer = None,    \
                                         preprocessor = None, \
                                         stop_words = None,   \
                                         max_features = 15000)

                bag_of_word = vectorizer.fit_transform(judged_review)


            elif docrepresentation == "BOW":
                # Initialize the "CountVectorizer" object, which is scikit-learn's
                # bag of words tool.
                print "Uisng Bag of Word"
                vectorizer = CountVectorizer(analyzer = "word",   \
                                             tokenizer = None,    \
                                             preprocessor = None, \
                                             stop_words = None,   \
                                             max_features = 15000)

                # fit_transform() does two functions: First, it fits the model
                # and learns the vocabulary; second, it transforms our training data
                # into feature vectors. The input to fit_transform should be a list of
                # strings.
                bag_of_word = vectorizer.fit_transform(judged_review)

            # Numpy arrays are easy to work with, so convert the result to an
            # array
            bag_of_word = bag_of_word.toarray()
            print bag_of_word.shape
            #vocab = vectorizer.get_feature_names()
            #print vocab

            # Sum up the counts of each vocabulary word
            #dist = np.sum(bag_of_word, axis=0)

            # For each, print the vocabulary word and the number of times it
            # appears in the training set
            #for tag, count in zip(vocab, dist):
            #    print count, tag

            print "Bag of word completed"

            X= bag_of_word
            y= relevance_label

            # print len(y)
            # print y
            numberOne = y.count(1)
            # print "Number of One", numberOne

            numberZero = y.count(0)
            print "Number of One", numberOne
            print "Number of Zero", numberZero
            datasize = len(X)
            prevelance = (numberOne * 1.0) / datasize
            # print "Number of zero", numberZero


            X = pd.DataFrame(bag_of_word)
            y = pd.Series(relevance_label)

            #This stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.
            #For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=test_size, stratify=y)

            train_index_list = X_train.index.values
            test_index_list = X_test.index.values

            #print train_index_list
            #print y_train[1354]
            for indexer in train_index_list:
                #print "File:", indexer, ":", len(X_train.as_matrix()[indexer])
                docNo = docIndex_DocNo[indexer]
                pred_str = pred_str + str(topic)+" "+str(docNo) + " " + str(y_train[indexer]) + "\n"
                #print pred_str

            #exit(0)
            '''print y_test
            print X_test.index.values[0]
            print X_train.index.values[0]
            print y_train[X_train.index.values[0]]
            '''
            print "=========Before Sampling======"

            print "Whole Dataset size: ", datasize
            print "Number of Relevant", numberOne
            print "Number of non-relevant", numberZero
            print "prevelance ratio", prevelance * 100
            print "Test size :", len(y_test)
            print "Train size :", len(y_train)

            #print "After", y_train

            print "=========After Sampling======"
            #print "Whole Dataset size: ", datasize
            print "Test size :", len(y_test)
            print "Train size :", len(y_train)


            if protocol == 'Basic':
                print '----Started Training----'
                model = LogisticRegression()
                model.fit(X_train, y_train)
            else:
                print '----Started Training----'
                model = LogisticRegression()
                size = len(X_train) - n_labeled

                if size<0:
                    print "Train Size:", len(X_train) , "seed:", n_labeled
                    size = len(X_train)

                numberofloop = size / batch_size
                if numberofloop == 0:
                    numberofloop = 1
                print "Number of loop",numberofloop

                X_train = X_train.as_matrix()

                #initial_X_train = X_train[:n_labeled]
                #initial_y_train = y_train[:n_labeled]

                initial_X_train = []
                initial_y_train = []

                # collecting the seed list from the Rankers
                seed_list = Ranker_topic_to_doclist[topic]
                seed_counter = 0
                seed_one_counter = 0
                seed_zero_counter = 0
                ask_for_label = 0
                for seed_counter in xrange(0,n_labeled):
                    documentNumber = seed_list[seed_counter]
                    if documentNumber not in docNo_docIndex:
                        continue
                    index = docNo_docIndex[documentNumber]
                    #we are making sure that this document doea not belong to test set
                    if index not in test_index_list:
                        labelValue = int(docNo_label[documentNumber])
                        ask_for_label = ask_for_label + 1
                        listIndex =  train_index_list.tolist().index(index)
                        #print "Found in List Index number", listIndex
                        initial_X_train.append(X_train[listIndex])
                        initial_y_train.append(labelValue)
                        if labelValue == 1:
                            seed_one_counter = seed_one_counter + 1
                        if labelValue == 0:
                            seed_zero_counter = seed_zero_counter + 1


                if seed_one_counter == 0:
                    print "No Relevant Document found in Seed set for topic", topic
                    seed_counter = n_labeled
                    while seed_one_counter < 2:

                        documentNumber = seed_list[seed_counter]
                        print "Seed Counter", seed_counter
                        seed_counter = seed_counter + 1

                        if documentNumber not in docNo_docIndex:
                            continue
                        index = docNo_docIndex[documentNumber]
                        # we are making sure that this document doea not belong to test set
                        if index not in test_index_list:
                            labelValue = int(docNo_label[documentNumber])
                            ask_for_label = ask_for_label + 1
                            # we are skipping label 0 because we do not need it
                            if labelValue == 0:
                                continue
                            listIndex = train_index_list.tolist().index(index)
                            # print "Found in List Index number", listIndex
                            initial_X_train.append(X_train[listIndex])
                            initial_y_train.append(labelValue)
                            if labelValue == 1:
                                seed_one_counter = seed_one_counter + 1
                            if labelValue == 0:
                                seed_zero_counter = seed_zero_counter + 1


                print "Number of human labels:", ask_for_label, "# relevant document in seed set", seed_one_counter

                if seed_zero_counter == 0:
                    print "No Non-Relevant Document found in Seed set for topic", topic
                    seed_counter = n_labeled
                    while seed_zero_counter < 2:

                        documentNumber = seed_list[seed_counter]
                        print "Seed Counter", seed_counter
                        seed_counter = seed_counter + 1

                        if documentNumber not in docNo_docIndex:
                            continue
                        index = docNo_docIndex[documentNumber]
                        # we are making sure that this document doea not belong to test set
                        if index not in test_index_list:
                            labelValue = int(docNo_label[documentNumber])
                            ask_for_label = ask_for_label + 1
                            # we are skipping label 1 because we do not need it
                            if labelValue == 1:
                                continue
                            listIndex = train_index_list.tolist().index(index)
                            # print "Found in List Index number", listIndex
                            initial_X_train.append(X_train[listIndex])
                            initial_y_train.append(labelValue)
                            if labelValue == 1:
                                seed_one_counter = seed_one_counter + 1
                            if labelValue == 0:
                                seed_zero_counter = seed_zero_counter + 1
                print "Number of human labels:", ask_for_label, "# non-relevant document in seed set", seed_zero_counter
                #print type(initial_X_train)

                if sampling == True:
                    print "Oversampling in the seed list"
                    #ros = RandomOverSampler()
                    #ros = RandomUnderSampler()
                    #ros = SMOTE(random_state=42)
                    ros = ADASYN()
                    initial_X_train_sampled, initial_y_train_sampled = ros.fit_sample(initial_X_train, initial_y_train)
                    initial_X_train = initial_X_train_sampled
                    initial_y_train = initial_y_train_sampled

                    initial_X_train = initial_X_train.tolist()
                    initial_y_train = initial_y_train.tolist()

                initial_X_test = X_train[n_labeled:]
                initial_y_test = y_train[n_labeled:]

                initial_X_test = initial_X_test.tolist()
                initial_y_test = initial_y_test.tolist()

                print "Before Loop Lenght:", len(initial_X_train), len(initial_y_train)
                predictableSize = len(initial_X_test)
                isPredictable = [1] * predictableSize  # initially we will predict all

                loopCounter = 0
                while loopCounter<numberofloop:
                    print "Loop:", loopCounter
                    loopDocList = []

                    if protocol == 'SPL':
                        model = LogisticRegression()
                    model.fit(initial_X_train, initial_y_train)

                    # here is queueSize is the number of predictable element
                    queueSize = isPredictable.count(1)

                    if protocol == 'CAL':
                        print "####CAL####"
                        queue = Queue.PriorityQueue(queueSize)
                        y_prob = []
                        counter = 0
                        for counter in xrange(0,predictableSize):
                            if isPredictable[counter] == 1:
                                # reshapping reshape(1,-1) because it does not take one emelemt array
                                # list does not contain reshape so we are using np,array
                                # model.predit returns two value in index [0] of the list
                                y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1,-1))[0]
                                #y_prob = model.predict(initial_X_test[counter])
                                #print y_prob
                                queue.put(relevance(y_prob[1], counter))


                        batch_counter = 0
                        while not queue.empty():
                            if batch_counter == batch_size:
                                break
                            item = queue.get()
                            #print len(item)
                            #print item.priority, item.index

                            isPredictable[item.index] = 0 # not predictable
                            initial_X_train.append(initial_X_test[item.index])
                            initial_y_train.append(initial_y_test[item.index])
                            #print "Docs:", initial_X_test[item.index]
                            loopDocList.append(int(initial_y_test[item.index]))
                            batch_counter = batch_counter + 1
                            #print X_train.append(X_test.pop(item.priority))

                    if protocol == 'SAL':
                        print "####SAL####"
                        queue = Queue.PriorityQueue(queueSize)
                        y_prob = []
                        counter = 0
                        for counter in xrange(0,predictableSize):
                            if isPredictable[counter] == 1:
                                # reshapping reshape(1,-1) because it does not take one emelemt array
                                # list does not contain reshape so we are using np,array
                                # model.predit returns two value in index [0] of the list
                                y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1,-1))[0]
                                entropy = (-1)*(y_prob[0]*log(y_prob[0],2)+y_prob[1]*log(y_prob[1],2))
                                queue.put(relevance(entropy, counter))


                        batch_counter = 0
                        while not queue.empty():
                            if batch_counter == batch_size:
                                break
                            item = queue.get()
                            #print len(item)
                            #print item.priority, item.index
                            isPredictable[item.index] = 0 # not predictable
                            initial_X_train.append(initial_X_test[item.index])
                            initial_y_train.append(initial_y_test[item.index])
                            loopDocList.append(int(initial_y_test[item.index]))
                            batch_counter = batch_counter + 1
                            #print X_train.append(X_test.pop(item.priority))



                    if protocol == 'SPL':
                        print "####SPL####"
                        randomArray =[]
                        randomArrayIndex = 0
                        for counter in xrange(0, predictableSize):
                            if isPredictable[counter] == 1:
                                randomArray.append(counter)
                                randomArrayIndex = randomArrayIndex + 1
                        import random
                        #random.shuffle(randomArray)

                        batch_counter = 0
                        for batch_counter in xrange(0,batch_size):
                            itemIndex = randomArray[batch_counter]
                            isPredictable[itemIndex] = 0
                            initial_X_train.append(initial_X_test[itemIndex])
                            initial_y_train.append(initial_y_test[itemIndex])
                            loopDocList.append(int(initial_y_test[itemIndex]))



                    loopCounter = loopCounter + 1
                    y_pred = model.predict(X_test)
                    f1score = f1_score(y_test, y_pred, average='binary')
                    precision = precision_score(y_test, y_pred, average='binary')
                    recall = recall_score(y_test, y_pred, average='binary')
                    print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred), "Number of predicted (0):", np.prod(y_pred.shape) - np.count_nonzero(y_pred)
                    '''
                    precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

                    recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

                    So now you are dividing 0/0.'''
                    if f1score > best_f1:
                        best_f1 = f1score
                    print "Iteration: ", loopCounter," Added # of Relevant Docs:", loopDocList.count(1), " Added # of Non-relevant Docs:", loopDocList.count(0)
                    print "New Train Length:", len(initial_X_train)  # len(initial_y_train)

            y_pred = model.predict(X_test)

            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1score = f1_score(y_test, y_pred, average='binary')
            if f1score > best_f1:
                best_f1 = f1score
            index=0
            #for y in y_pred:
                #print 'real:', y_test[index], 'pred:', y
            #    index=index+1
            print "precision score:", precision
            print "recall score:", recall
            print "f-1 score:", f1score

            num_correct = (y_pred == y_test).sum()
            recall1 = (num_correct*1.0) / len(y_test)
            print "model accuracy (%): ", recall1 * 100, "%"

            s = s + topic + "," + str(datasize) + "," + str(numberOne) + "," + str(numberZero) + "," + str(prevelance) + "," + str(precision) + "," + str(recall) + "," + str(f1score) + "," + str(recall1) + "," + str(best_f1) +"\n";
            # writing the actual and prediction



            counter = 0
            for docIndex in test_index_list:
                docNo = docIndex_DocNo[docIndex]
                pred_str = pred_str + str(topic) + " "+str(docNo) + " " + str(y_pred[counter]) + "\n"
                counter = counter + 1

        text_file = open(result_location, "w")
        text_file.write(s)
        text_file.close()

        text_file = open(predicted_location, "w")
        text_file.write(pred_str)
        text_file.close()




