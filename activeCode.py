'''Active learning for labeling the relevant document for TREC-8 dataset
@author: Md Mustafizur Rahman (nahid@utexas.edu)'''

import os
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
#import nltk
#import copy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#from libact.base.dataset import Dataset, import_libsvm_sparse
#from libact.models import *
#from libact.query_strategies import *
#from libact.labelers import IdealLabeler
from imblearn.over_sampling import RandomOverSampler
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
test_size = 0.2    # the percentage of samples in the dataset that will be
test_size_set = [0.2, 0.4, 0.6, 0.8]
#test_size_set = [0.7, 0.8]

datasource = sys.argv[1] # can be  dataset = ['TREC8', 'gov2', 'WT']
n_labeled =  int(sys.argv[2]) #50      # number of samples that are initially labeled
batch_size = int(sys.argv[3]) #50
protocol = sys.argv[4] #'SAL' can be ['SAL', 'CAL', 'SPL']
preloaded = True

processed_file_location = ''
start_topic = 0
end_topic = 0

result_location=''
predicted_location='/work/04549/mustaf/wrangler/data/TREC/WT2014/prediction/'

if datasource=='TREC8':
    processed_file_location = '/work/04549/mustaf/wrangler/data/TREC/TREC8/processed.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/wrangler/data/TREC/TREC8/relevance.txt'
    result_location = '/work/04549/mustaf/wrangler/data/TREC/TREC8/result/'
    predicted_location = '/work/04549/mustaf/wrangler/data/TREC/TREC8/prediction/'
    start_topic = 401
    end_topic = 451
elif datasource=='gov2':
    processed_file_location = '/work/04549/mustaf/wrangler/data/TREC/gov2/processed.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/wrangler/data/TREC/gov2/qrels.tb06.top50.txt'
    result_location = '/work/04549/mustaf/wrangler/data/TREC/gov2/result/'
    predicted_location = '/work/04549/mustaf/wrangler/data/TREC/gov2/prediction/'
    start_topic = 801
    end_topic = 851
elif datasource=='WT2013':
    processed_file_location = '/work/04549/mustaf/wrangler/data/TREC/WT2013/processed_new.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/wrangler/data/TREC/WT2013/qrelsadhoc2013.txt'
    result_location = '/work/04549/mustaf/wrangler/data/TREC/WT2013/result/'
    predicted_location = '/work/04549/mustaf/wrangler/data/TREC/WT2013/prediction/'
    start_topic = 201
    end_topic = 251
else:
    processed_file_location = '/work/04549/mustaf/wrangler/data/TREC/WT2013/processed_new.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/wrangler/data/TREC/WT2014/qrelsadhoc2014.txt'
    result_location = '/work/04549/mustaf/wrangler/data/TREC/WT2014/result/'
    predicted_location = '/work/04549/mustaf/wrangler/data/TREC/WT2014/prediction/'
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


for test_size in test_size_set:
    seed = 1335
    for fold in xrange(1,6):
        np.random.seed(seed)
        seed = seed + fold
        result_location_final = result_location + str(
            test_size) + '_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location_final = predicted_location + str(
            test_size) + '_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'

        s = "";
        pred_str = ""
        #for topic in sorted(topic_to_doclist.keys()):
        for topic in xrange(start_topic,end_topic):
            print "Topic:", topic
            if topic == 202:
                continue
            topic = str(topic)

            topic_to_doclist = {}  # key is the topic(string) and value is the list of docNumber
            docNo_label = {}  # key is the DocNo and the value is the label
            docIndex_DocNo = {} # key is the index used in my code value is the actual DocNo
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
                    docIndex = docIndex + 1
                    judged_review.append(all_reviews[documentNo])
                    relevance_label.append(docNo_label[documentNo])


            if docrepresentation == "TF-IDF":
                print "Using TF-IDF"
                vectorizer = TfidfVectorizer(min_df=5, \
                                         analyzer = "word",   \
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

            #print y_train[1354]
            for indexer in train_index_list:
                docNo = docIndex_DocNo[indexer]
                pred_str = pred_str + str(docNo) + " " + str(y_train[indexer]) + "\n"
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

            if sampling == True:
                ros = RandomOverSampler()
                X_train, y_train = ros.fit_sample(X_train, y_train)
                X_train = X_train.tolist()
                y_train = y_train.tolist()

                if y_train.count(1)== 0:
                    print topic, "ZERO"

                #print "Before", y_train
                #print "Number of one in train after sampling", y_train.count(1)
                #print "Number of one in test after sampling", y_test.count(1)

                # we have to do this because randomoversampling placing all the zero at the first halh
                # and all the one label at last half
                # which is creating problem for activer learning (logistic regression module)
                # we are passing the first 10 sample and because of this the first ten sample
                # only contains zero

                X_a, X_b, y_a, y_b = \
                    train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)

                X_train = X_a + X_b
                y_train = y_a + y_b

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
                #numberofloop can be zero lenX_train = 138 and seed =70 size= 68 and batch_size = 100
                if numberofloop == 0:
                    numberofloop = 1

                print "Number of loop",numberofloop


                initial_X_train = X_train[:n_labeled]
                initial_y_train = y_train[:n_labeled]

                initial_X_test = X_train[n_labeled:]
                initial_y_test = y_train[n_labeled:]

                print "Before Loop Lenght:", len(initial_X_train), len(initial_y_train)
                predictableSize = len(initial_X_test)
                isPredictable = [1] * predictableSize  # initially we will predict all

                loopCounter = 0
                while loopCounter<numberofloop:
                    print "Loop:", loopCounter

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
                        random.shuffle(randomArray)

                        batch_counter = 0
                        for batch_counter in xrange(0,batch_size):
                            itemIndex = randomArray[batch_counter]
                            isPredictable[itemIndex] = 0
                            initial_X_train.append(initial_X_test[itemIndex])
                            initial_y_train.append(initial_y_test[itemIndex])



                    print "Inside Loop Lenght:", len(initial_X_train), len(initial_y_train)
                    loopCounter = loopCounter + 1


            y_pred = model.predict(X_test)

            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1score = f1_score(y_test, y_pred, average='binary')

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

            s = s + topic + "," + str(datasize) + "," + str(numberOne) + "," + str(numberZero) + "," + str(prevelance) + "," + str(precision) + "," + str(recall) + "," + str(f1score) + "," + str(recall1) + "\n";
            # writing the actual and prediction



            counter = 0
            for docIndex in test_index_list:
                docNo = docIndex_DocNo[docIndex]
                pred_str = pred_str + str(docNo) + " " + str(y_pred[counter]) + "\n"
                counter = counter + 1

        text_file = open(result_location_final, "w")
        text_file.write(s)
        text_file.close()

        text_file = open(predicted_location_final, "w")
        text_file.write(pred_str)
        text_file.close()




