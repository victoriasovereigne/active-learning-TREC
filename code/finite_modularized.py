'''Active learning for labeling the relevant document for TREC-8 dataset
@author: Md Mustafizur Rahman (nahid@utexas.edu)'''

import os
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
import math
import nltk
import copy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsemble
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
# import pandas as pd
import Queue

import logging
logging.basicConfig()

from undersampler import *
from protocol import *
from helper import *

TEXT_DATA_DIR = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/'
RELEVANCE_DATA_DIR = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/TREC8/relevance.txt'
docrepresentation = "TF-IDF"  # can be BOW, TF-IDF
sampling=False # can be True or False
command_prompt_use = True

#if command_prompt_use == True:
datasource = 'TREC8' #sys.argv[1] # can be  dataset = ['TREC8', 'gov2', 'WT']
protocol = 'SAL' #sys.argv[2]
use_ranker = 'False' #sys.argv[3]
iter_sampling = 'True' #sys.argv[4]
correction = 'False' #sys.argv[5] #'SAL' can be ['SAL', 'CAL', 'SPL']
train_per_centage_flag = 'True' #sys.argv[6]
under_sampling = 'False' #sys.argv[7]

#parameter set # all FLAGS must be string
'''
datasource = 'WT2014'  # can be  dataset = ['TREC8', 'gov2', 'WT']
protocol = 'CAL'  # 'SAL' can be ['SAL', 'CAL', 'SPL']
use_ranker = 'True'
iter_sampling = 'True'
correction = 'False'
train_per_centage_flag = 'True'
'''
print "Ranker_use", use_ranker
print "iter_sampling", iter_sampling
print "correction", correction
print "train_percenetae", train_per_centage_flag
print "under sampling", under_sampling


test_size = 0    # the percentage of samples in the dataset that will be
#test_size_set = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
test_size_set = [0.2]
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

ranker_location = {}
ranker_location["WT2013"] = "/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/systemRanking/WT2013/input.ICTNET13RSR2"
ranker_location["WT2014"] = "/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/systemRanking/WT2014/input.Protoss"
ranker_location["gov2"] = "/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/systemRanking/gov2/input.indri06AdmD"
ranker_location["TREC8"] = "/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/systemRanking/TREC8/input.ibmg99b"

n_labeled =  10 #50      # number of samples that are initially labeled
batch_size = 25 #50
preloaded = True
topicSkipList = [202,225,255,278,805]

skipList = []
topicBucketList = []
processed_file_location = ''
start_topic = 0
end_topic = 0

base_address = "/u/vlestari/Documents/Summer/IR/result/"

# base_address = "/home/nahid/UT_research/clueweb12/nooversample_result1/"

base_address = base_address +str(datasource)+"/"
if use_ranker == 'True':
    base_address = base_address + "ranker/"
    use_ranker = True
else:
    base_address = base_address + "no_ranker/"
    use_ranker = False
if iter_sampling == 'True':
    base_address = base_address + "oversample/"
    iter_sampling = True
if iter_sampling == 'False':
    iter_sampling = False
if correction == 'True':
    base_address = base_address + "htcorrection/"
    correction = True
if correction == 'False':
    correction = False

# --------------------------------
if under_sampling == 'True':
    base_address = base_address + "undersample/"
    under_sampling = True
if under_sampling == 'False':
    under_sampling = False
# --------------------------------

if train_per_centage_flag == 'True':
    train_per_centage_flag = True
else:
    train_per_centage_flag = False

print "base address:", base_address


if iter_sampling == True and correction == True:
    print "Over sampling and HT correction cannot be done together"
    exit(-1)
# --------------------------------
if iter_sampling == True and under_sampling == True:
    print "Over sampling and under sampling cannot be done together"
    exit(-1)

if datasource=='TREC8':
    processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/TREC8/processed.txt'
    RELEVANCE_DATA_DIR = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/TREC8/relevance.txt'
    start_topic = 401
    end_topic = 451
elif datasource=='gov2':
    processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/gov2/processed.txt'
    RELEVANCE_DATA_DIR = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/gov2/qrels.tb06.top50.txt'
    start_topic = 801
    end_topic = 851
elif datasource=='WT2013':
    processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/WT2013/processed_new.txt'
    RELEVANCE_DATA_DIR = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/WT2013/qrelsadhoc2013.txt'
    start_topic = 201
    end_topic = 251
else:
    processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/WT2014/processed_new.txt'
    RELEVANCE_DATA_DIR = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dataAll/IndriProcessedData/WT2014/qrelsadhoc2014.txt'
    start_topic = 251
    end_topic = 301


all_reviews = {}
learning_curve = {} # per batch value for  validation set
learning_curve_precision = {}
learning_curve_recall = {}
learning_curve_sensitivity = {}
learning_curve_specificity = {}

if preloaded==False:
    all_reviews = load_pickle(TEXT_DATA_DIR, processed_file_location)

else:
    input = open(processed_file_location, 'rb')
    all_reviews = pickle.load(input)
    print "pickle loaded"

Ranker_topic_to_doclist = get_ranker(ranker_location, datasource)

for test_size in test_size_set:
    seed = 1335
    for fold in xrange(1,2):
        np.random.seed(seed)
        seed = seed + fold
        result_location = base_address + 'result_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location_base = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_'
        human_label_location = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_'

        learning_curve_location = base_address + 'learning_curve_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'

        s = "";
        pred_str = ""
        #for topic in sorted(topic_to_doclist.keys()):
        for topic in xrange(start_topic,end_topic):
            print "Topic:", topic
            if topic in topicSkipList:
                print "Skipping Topic :", topic
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
            for lines in f:
                values = lines.split()
                topicNo = values[0]
                if topicNo != topic:
                    continue
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

            docList = topic_to_doclist[topic]
            print 'number of documents', len(docList)
            relevance_label = []
            judged_review = []

            docIndex = 0
            for documentNo in docList:
                if all_reviews.has_key(documentNo):
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
            print "Bag of word completed"

            X= bag_of_word
            y= relevance_label

            numberOne = y.count(1)
            numberZero = y.count(0)
            print "Number of One", numberOne
            print "Number of Zero", numberZero
            datasize = len(X)
            prevelance = (numberOne * 1.0) / datasize
         
            print "=========Before Sampling======"
            print "Whole Dataset size: ", datasize
            print "Number of Relevant", numberOne
            print "Number of non-relevant", numberZero
            print "prevelance ratio", prevelance * 100

            print '----Started Training----'
            model = LogisticRegression()
            size = len(X) - n_labeled
            num_subsets = 3
            ens = EnsembleClassifier(num_subsets)

            if size<0:
                print "Train Size:", len(X) , "seed:", n_labeled
                size = len(X)

            initial_X_train = []
            initial_y_train = []
            train_index_list = []

            initial_X_agg = []
            initial_y_agg = []

            # collecting the seed list from the Rankers
            seed_list = Ranker_topic_to_doclist[topic]
            seed_counter = 0
            seed_one_counter = 0
            seed_zero_counter = 0
            ask_for_label = 0

            # ==============================================================================================
            if use_ranker == True:
                loopCounter = 0

                seed_size_limit = math.ceil(train_per_centage[loopCounter] * len(X))
                print "INitial Seed Limit", seed_size_limit
                seed_start = 0
                seed_counter = 0
                while True:
                    print "seed size limit:", seed_size_limit
                    while seed_counter < seed_size_limit:
                        documentNumber = seed_list[seed_counter]
                        seed_counter = seed_counter + 1
                        if documentNumber not in docNo_docIndex:
                            continue
                        index = docNo_docIndex[documentNumber]
                        train_index_list.append(index)
                        labelValue = int(docNo_label[documentNumber])
                        ask_for_label = ask_for_label + 1
                        initial_X_train.append(X[index])
                        initial_y_train.append(labelValue)

                        if labelValue == 1:
                            seed_one_counter = seed_one_counter + 1
                        if labelValue == 0:
                            seed_zero_counter = seed_zero_counter + 1

                        #print seed_one_counter, seed_zero_counter
                    if seed_zero_counter == 0 or seed_one_counter == 0:
                        print "Seed Size Limit:", seed_size_limit, "Seed Explored so far:", seed_counter
                        print seed_one_counter, seed_zero_counter
                        ####################################Store Result###################
                        y_pred_all = {}
                        human_label_location_final, human_label_str = write_human_label_location(train_index_list, y, y_pred_all, 
                            docIndex_DocNo, topic, human_label_location, train_per_centage[loopCounter])
                        print human_label_location_final

                        for train_index in xrange(0, len(X)):
                            if train_index not in train_index_list:
                                y_pred_all[train_index] = 0 # consider all are non-relevants

                        y_pred = []
                        for key, value in y_pred_all.iteritems():
                            y_pred.append(value)

                        ##################
                        write_predicted_location(X, docIndex_DocNo, topic, y_pred_all, predicted_location_base, 
                                train_per_centage[loopCounter])
                        f1score = get_f1score(y, y_pred, learning_curve, train_per_centage[loopCounter])

                        precision = get_precision(y, y_pred, learning_curve_precision, train_per_centage[loopCounter])

                        recall = get_recall(y, y_pred, learning_curve_recall, train_per_centage[loopCounter])

                        print "Score in non-active stage"
                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        loopCounter = loopCounter + 1
                        seed_size_limit = math.ceil(train_per_centage[loopCounter] * len(X))
                    if seed_zero_counter > 0 and seed_one_counter > 0:
                        print "Requirement made at counter", loopCounter,  "(1:)", seed_one_counter, "(0):", seed_zero_counter
                        print "Seed Size Limit:", seed_size_limit, "Seed Explored so far:", seed_counter
                        break

                unmodified_train_X = copy.deepcopy(initial_X_train)
                unmodified_train_y = copy.deepcopy(initial_y_train)
                sampling_weight = []

                for sampling_index in xrange(0, len(initial_X_train)):
                    sampling_weight.append(1.0)

                # Ranker needs oversampling, but when HTCorrection true we cannot perform oversample
                if use_ranker == True and correction == False:
                    initial_X_train_sampled, initial_y_train_sampled = update_initial_train(iter_sampling, 
                        under_sampling, initial_X_train, initial_y_train, num_subsets)

                    if under_sampling == True:
                        initial_X_agg = initial_X_train_sampled.tolist()
                        initial_y_agg = initial_y_train_sampled.tolist()
                    else:
                        initial_X_train = initial_X_train_sampled.tolist()
                        initial_y_train = initial_y_train_sampled.tolist()

                # fill the agg
                # initial_X_agg, initial_y_agg = fill_agg(initial_X_train, initial_y_train, num_subsets)
                # print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"
                # print np.array(initial_X_agg).shape
                # print np.array(initial_X_train).shape
                # print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"

                initial_X_test = []
                initial_y_test = []
                test_index_list = create_test_index_list(X, y, train_index_list, initial_X_test, initial_y_test)

                print "Before Loop Lenght:", len(initial_X_train), len(initial_y_train)
                predictableSize = len(initial_X_test)
                isPredictable = [1] * predictableSize  # initially we will predict all

                #loopCounter = 0
                best_model = 0
                learning_batch_size = n_labeled  # starts with the seed size

                # ==============================================================
                # use ranker == True
                # train percentage == False
                # ==============================================================
                if train_per_centage_flag == False:
                    numberofloop = math.ceil(size / batch_size)
                    if numberofloop == 0:
                        numberofloop = 1
                    print "Number of loop", numberofloop

                    # start while
                    while loopCounter <= numberofloop:
                        print "Loop:", loopCounter
                        loopDocList = []

                        # new method fit model
                        fit_model(model, ens, protocol, False, under_sampling, 
                                    initial_X_train, initial_y_train, sampling_weight,
                                    initial_X_agg, initial_y_agg)

                        y_pred_all = {}

                        for train_index in train_index_list:
                            y_pred_all[train_index] = y[train_index]
                        
                        y_pred_all = predict_y_pred_all(X, train_index_list, y_pred_all, model, ens, under_sampling)
                        y_pred = []

                        for key, value in y_pred_all.iteritems():
                            y_pred.append(value)

                        f1score = get_f1score(y, y_pred, learning_curve, train_per_centage[loopCounter])

                        precision = get_precision(y, y_pred, learning_curve_precision, train_per_centage[loopCounter])

                        recall = get_recall(y, y_pred, learning_curve_recall, train_per_centage[loopCounter])

                        learning_batch_size = learning_batch_size + batch_size

                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        if isPredictable.count(1) == 0:
                            break

                        '''
                        precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.
                        recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.
                        So now you are dividing 0/0.'''
                        # here is queueSize is the number of predictable element
                        queueSize = isPredictable.count(1)

                        if protocol == 'CAL':
                            queue = CAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test)
                            empty_queue(queue, 0, batch_size, isPredictable, correction, 
                                        unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test, 
                                        sampling_weight, train_index_list, test_index_list, loopDocList)

                        if protocol == 'SAL':
                            queue = SAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test)
                            empty_queue(queue, 0, batch_size, isPredictable, correction, 
                                        unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test, 
                                        sampling_weight, train_index_list, test_index_list, loopDocList)

                        if protocol == 'SPL':
                            randomArray = SPL_shuffle_array(predictableSize, isPredictable)
                            SPL_train_percentage_false(batch_size, randomArray, isPredictable, 
                                unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test,
                                sampling_weight, train_index_list, test_index_list, loopDocList)

                        initial_X_train, initial_y_train = update_initial_train(iter_sampling, under_sampling, 
                                                            unmodified_train_X, unmodified_train_y, num_subsets)  

                        loopCounter = loopCounter + 1
                    # end while
                # ==============================================================
                # use ranker == True
                # train percentage == True
                # ANOTHER ONE
                # ==============================================================
                else:
                    print "Loop Counter after seed:", loopCounter
                    numberofloop = len(train_per_centage)
                    train_size_controller = len(unmodified_train_X)

                    while loopCounter < numberofloop:
                        size_limit = math.ceil(train_per_centage[loopCounter]*len(X))

                        print "Loop:", loopCounter
                        print "Initial size:",train_size_controller, "limit:", size_limit

                        loopDocList = []

                        # new method fit model
                        fit_model(model, ens, protocol, correction, under_sampling, 
                                    initial_X_train, initial_y_train, sampling_weight,
                                    initial_X_agg, initial_y_agg)

                        y_pred_all = get_prediction_y_pred(train_index_list, docIndex_DocNo, topic, human_label_location, 
                            train_per_centage, loopCounter, under_sampling, ens, model, X, y)

                        ##################
                        y_pred = write_pred_to_file(y_pred_all, X, docIndex_DocNo, topic, predicted_location_base, train_per_centage, loopCounter)
                        f1score = get_f1score(y, y_pred, learning_curve, train_per_centage[loopCounter])

                        precision = get_precision(y, y_pred, learning_curve_precision, train_per_centage[loopCounter])

                        recall = get_recall(y, y_pred, learning_curve_recall, train_per_centage[loopCounter])

                        learning_batch_size = learning_batch_size + batch_size

                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        sensitivity, specificity = get_sensitivity_specificity(y, y_pred, learning_curve_specificity, train_per_centage[loopCounter])
                        learning_batch_size = learning_batch_size + batch_size
                        
                        print "sensitivity:", sensitivity
                        print "specificity:", specificity

                        if isPredictable.count(1) == 0:
                            break

                        '''
                        precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.
                        recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.
                        So now you are dividing 0/0.'''
                        # here is queueSize is the number of predictable element
                        queueSize = isPredictable.count(1)

                        if protocol == 'CAL':
                            queue = CAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test)
                            train_size_controller = empty_queue(queue, train_size_controller, size_limit, isPredictable, correction, 
                                        unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test, 
                                        sampling_weight, train_index_list, test_index_list, loopDocList)

                        if protocol == 'SAL':
                            queue = SAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test)
                            train_size_controller = empty_queue(queue, train_size_controller, size_limit, isPredictable, correction, 
                                        unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test, 
                                        sampling_weight, train_index_list, test_index_list, loopDocList)

                        if protocol == 'SPL':
                            randomArray = SPL_shuffle_array(predictableSize, isPredictable)
                            train_size_controller = SPL_train_percentage_true(train_size_controller, size_limit, randomArray, 
                                                        isPredictable, unmodified_train_X, unmodified_train_y, 
                                                        initial_X_test, initial_y_test, sampling_weight, 
                                                        train_index_list, test_index_list, loopDocList)
                        initial_X_train, initial_y_train = update_initial_train(iter_sampling, under_sampling, 
                                                            unmodified_train_X, unmodified_train_y, num_subsets)
                        
                        if under_sampling == True:
                            initial_X_agg = initial_X_train
                            initial_y_agg = initial_y_train
                        
                        loopCounter = loopCounter + 1

                y_pred_all = {}
                human_label_location_final, human_label_str = write_human_label_location(train_index_list, y, y_pred_all, 
                            docIndex_DocNo, topic, human_label_location, 1.1)
                y_pred_all = predict_y_pred_all(X, train_index_list, y_pred_all, model, ens, under_sampling)

                y_pred = []
                for key, value in y_pred_all.iteritems():
                    y_pred.append(value)

                f1score = f1_score(y, y_pred, average='binary')
                precision = precision_score(y, y_pred, average='binary')
                recall = recall_score(y, y_pred, average='binary')

                if train_per_centage_flag == True:
                    if (learning_curve.has_key(1.1)):
                        tmplist = learning_curve.get(1.1)
                        tmplist.append(f1score)
                        learning_curve[1.1] = tmplist
                    else:
                        tmplist = []
                        tmplist.append(f1score)
                        learning_curve[1.1] = tmplist

                    write_predicted_location(X, docIndex_DocNo, topic, y_pred_all, 
                        predicted_location_base, 1.1)

                print "precision score:", precision
                print "recall score:", recall
                print "f-1 score:", f1score

                # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
                '''
                precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.
                recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.
                So now you are dividing 0/0.'''

                accuracy = accuracy_score(y, y_pred)
                print "model accuracy (%): ", accuracy * 100, "%"

                s = s + topic + "," + str(datasize) + "," + str(numberOne) + "," + str(numberZero) + "," + str(
                    prevelance) + "," + str(precision) + "," + str(recall) + "," + str(f1score) + "," + str(
                    accuracy) + "," + str(best_f1) + "\n";
                # writing the actual and prediction

                counter = 0
                # doing predcition again for all documents in the validation and test set for writing the prediction files
                
                for docIndex in xrange(0, len(X)):
                    docNo = docIndex_DocNo[docIndex]
                    pred_str = pred_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"
                    counter = counter + 1

                text_file = open(result_location, "w")
                text_file.write(s)
                text_file.close()

                text_file = open(predicted_location, "w")
                text_file.write(pred_str)
                text_file.close()

            # ==============================================================
            # use_ranker == false
            # ==============================================================
            else:
                initial_X_train, initial_y_train = create_initial_training_set(X, y, seed_one_counter, n_labeled/2, 
                    train_index_list, initial_X_train, initial_y_train, 1)

                initial_X_train, initial_y_train = create_initial_training_set(X, y, seed_zero_counter, n_labeled/2, 
                    train_index_list, initial_X_train, initial_y_train, 0)

                print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"
                print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"
                print np.array(initial_X_train).shape, np.array(initial_y_train).shape
                print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"
                print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"

                # fill the agg
                initial_X_agg, initial_y_agg =  fill_agg(initial_X_train, initial_y_train, num_subsets)

                orig_X_agg = copy.deepcopy(initial_X_agg)

                if seed_zero_counter != seed_one_counter:
                    skipList.append(topic)

                unmodified_train_X = copy.deepcopy(initial_X_train)
                unmodified_train_y = copy.deepcopy(initial_y_train)
                sampling_weight = []

                for sampling_index in xrange(0, len(initial_X_train)):
                    sampling_weight.append(1.0)

                if sampling == True:
                    initial_X_train_sampled, initial_y_train_sampled = update_initial_train(iter_sampling, 
                        under_sampling, initial_X_train, initial_y_train, num_subsets)

                    initial_X_train = initial_X_train_sampled.tolist()
                    initial_y_train = initial_y_train_sampled.tolist()

                initial_X_test = []
                initial_y_test = []
                test_index_list = create_test_index_list(X, y, train_index_list, initial_X_test, initial_y_test)

                print "Before Loop Lenght:", len(initial_X_train), len(initial_y_train)
                predictableSize = len(initial_X_test)
                isPredictable = [1] * predictableSize  # initially we will predict all

                loopCounter = 0
                best_model = 0
                learning_batch_size = n_labeled # starts with the seed size

                # ==============================================================
                # use ranker == False
                # train percentage == False
                # ==============================================================
                if train_per_centage_flag == False:
                    numberofloop = math.ceil(size / batch_size)
                    if numberofloop == 0:
                        numberofloop = 1
                    print "Number of loop", numberofloop

                    # start while
                    while loopCounter <= numberofloop:
                        print "Loop:", loopCounter
                        loopDocList = []

                        if protocol == 'SPL':
                            model = LogisticRegression()

                        if correction == True:
                            model.fit(initial_X_train, initial_y_train, sample_weight=sampling_weight)
                        else:
                            model.fit(initial_X_train, initial_y_train)

                        y_pred_all = {}

                        for train_index in train_index_list:
                            y_pred_all[train_index] = y[train_index]

                        y_pred_all = predict_y_pred_all(X, train_index_list, y_pred_all, model, ens, under_sampling)
                        y_pred = []
                        for key, value in y_pred_all.iteritems():
                            y_pred.append(value)

                        f1score = f1_score(y, y_pred, average='binary')

                        if (learning_curve.has_key(learning_batch_size)):
                            tmplist = learning_curve.get(learning_batch_size)
                            tmplist.append(f1score)
                            learning_curve[learning_batch_size] = tmplist
                        else:
                            tmplist = []
                            tmplist.append(f1score)
                            learning_curve[learning_batch_size] = tmplist

                        learning_batch_size = learning_batch_size + batch_size
                        precision = precision_score(y, y_pred, average='binary')
                        recall = recall_score(y, y_pred, average='binary')

                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        if isPredictable.count(1) == 0:
                            break

                        '''
                        precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.
                        recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.
                        So now you are dividing 0/0.'''
                        # here is queueSize is the number of predictable element
                        queueSize = isPredictable.count(1)

                        if protocol == 'CAL':
                            queue = CAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test)
                            empty_queue(queue, 0, batch_size, isPredictable, correction, 
                                        unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test, 
                                        sampling_weight, train_index_list, test_index_list, loopDocList)

                        if protocol == 'SAL':
                            queue = SAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test)
                            empty_queue(queue, 0, batch_size, isPredictable, correction, 
                                        unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test, 
                                        sampling_weight, train_index_list, test_index_list, loopDocList)

                        if protocol == 'SPL':
                            randomArray = SPL_shuffle_array(predictableSize, isPredictable)
                            SPL_train_percentage_false(batch_size, randomArray, isPredictable, 
                                unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test,
                                sampling_weight, train_index_list, test_index_list, loopDocList)

                        initial_X_train, initial_y_train = update_initial_train(iter_sampling, under_sampling, 
                                                            unmodified_train_X, unmodified_train_y, num_subsets)

                        loopCounter = loopCounter + 1
                    # end while

                # ==============================================================
                # use ranker == False
                # train percentage == True
                # THIS ONE
                # ==============================================================
                else:
                    numberofloop = len(train_per_centage)
                    train_size_controller = len(initial_X_train)
                    
                    while loopCounter < numberofloop:
                        size_limit = math.ceil(train_per_centage[loopCounter]*len(X))

                        print "Loop:", loopCounter
                        print "Initial size:",train_size_controller, "limit:", size_limit

                        loopDocList = []

                        # new method fit model
                        fit_model(model, ens, protocol, correction, under_sampling, 
                                    initial_X_train, initial_y_train, sampling_weight, 
                                    initial_X_agg, initial_y_agg)

                        # need to update initial X, y agg
                        print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"
                        print (np.array_equal(initial_X_agg, orig_X_agg))
                        print (np.array_equal(initial_X_train, unmodified_train_X))
                        print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"

                        y_pred_all = get_prediction_y_pred(train_index_list, docIndex_DocNo, topic, human_label_location, 
                            train_per_centage, loopCounter, under_sampling, ens, model, X, y)

                        ##################
                        y_pred = write_pred_to_file(y_pred_all, X, docIndex_DocNo, topic, predicted_location_base, train_per_centage, loopCounter)
                        f1score = get_f1score(y, y_pred, learning_curve, train_per_centage[loopCounter])

                        precision = get_precision(y, y_pred, learning_curve_precision, train_per_centage[loopCounter])

                        recall = get_recall(y, y_pred, learning_curve_recall, train_per_centage[loopCounter])
                        
                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        sensitivity, specificity = get_sensitivity_specificity(y, y_pred, learning_curve_specificity, train_per_centage[loopCounter])
                        learning_batch_size = learning_batch_size + batch_size

                        print "sensitivity:", sensitivity
                        print "specificity:", specificity


                        if isPredictable.count(1) == 0:
                            break

                        '''
                        precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.
                        recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.
                        So now you are dividing 0/0.'''
                        # here is queueSize is the number of predictable element
                        queueSize = isPredictable.count(1)

                        if protocol == 'CAL':
                            queue = CAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test)
                            train_size_controller = empty_queue(queue, train_size_controller, size_limit, isPredictable, correction, 
                                        unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test, 
                                        sampling_weight, train_index_list, test_index_list, loopDocList)

                        if protocol == 'SAL':
                            queue = SAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test)
                            train_size_controller = empty_queue(queue, train_size_controller, size_limit, isPredictable, correction, 
                                        unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test, 
                                        sampling_weight, train_index_list, test_index_list, loopDocList)

                        if protocol == 'SPL':
                            randomArray = SPL_shuffle_array(predictableSize, isPredictable)
                            train_size_controller = SPL_train_percentage_true(train_size_controller, size_limit, randomArray, 
                                                        isPredictable, unmodified_train_X, unmodified_train_y, 
                                                        initial_X_test, initial_y_test, sampling_weight, 
                                                        train_index_list, test_index_list, loopDocList)
                        
                        # update initial train
                        initial_X_train, initial_y_train = update_initial_train(iter_sampling, under_sampling, 
                                                            unmodified_train_X, unmodified_train_y, num_subsets)

                        if under_sampling == True:
                            initial_X_agg = initial_X_train
                            initial_y_agg = initial_y_train

                        loopCounter = loopCounter + 1


                print "Fininshed loop", len(initial_X_train)
                y_pred_all = {}
                human_label_location_final, human_label_str = write_human_label_location(train_index_list, y, y_pred_all, 
                            docIndex_DocNo, topic, human_label_location, 1.1)

                y_pred_all = predict_y_pred_all(X, train_index_list, y_pred_all, model, ens, under_sampling)

                y_pred = []
                for key, value in y_pred_all.iteritems():
                    y_pred.append(value)

                f1score = get_f1score(y, y_pred, learning_curve, 1.1)
                precision = get_precision(y, y_pred, learning_curve_precision, 1.1)
                recall = get_recall(y, y_pred, learning_curve_recall, 1.1)

                f1score = f1_score(y, y_pred, average='binary')
                precision = precision_score(y, y_pred, average='binary')
                recall = recall_score(y, y_pred, average='binary')

                sensitivity, specificity = get_sensitivity_specificity(y, y_pred, learning_curve_specificity, 1.1)
                print "sensitivity:", sensitivity
                print "specificity:", specificity

                write_predicted_location(X, docIndex_DocNo, topic, y_pred_all, 
                        predicted_location_base, 1.1)
                '''
                precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.
                recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.
                So now you are dividing 0/0.'''


                accuracy = accuracy_score(y, y_pred)
                print "model accuracy (%): ", accuracy * 100, "%"

                s = s + topic + "," + str(datasize) + "," + str(numberOne) + "," + str(numberZero) + "," + str(prevelance) + "," + str(precision) + "," + str(recall) + "," + str(f1score) + "," + str(accuracy) + "," + str(best_f1) +"\n";
                # writing the actual and prediction
                counter = 0
                # doing predcition again for all documents in the validation and test set for writing the prediction files
                
                for docIndex in xrange(0, len(X)):
                    docNo = docIndex_DocNo[docIndex]
                    pred_str = pred_str + str(topic) + " "+str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"
                    counter = counter + 1

            text_file = open(result_location, "w")
            text_file.write(s)
            text_file.close()


for topic in skipList:
    print topic

text_file = open(learning_curve_location, "w")

s=""
for (key, valueList) in sorted(learning_curve.items()):
    size = len(valueList)
    sum = 0
    for value in valueList:
        sum = sum + value
    s = s + str(sum/size) + ","

text_file.write(s + '\n')

s=""
for (key, valueList) in sorted(learning_curve_precision.items()):
    size = len(valueList)
    sum = 0
    for value in valueList:
        sum = sum + value
    s = s + str(sum/size) + ","

text_file.write(s + '\n')

s=""
for (key, valueList) in sorted(learning_curve_recall.items()):
    size = len(valueList)
    sum = 0
    for value in valueList:
        sum = sum + value
    s = s + str(sum/size) + ","

text_file.write(s + '\n')

s=""
for (key, valueList) in sorted(learning_curve_specificity.items()):
    size = len(valueList)
    sum = 0
    for value in valueList:
        sum = sum + value
    s = s + str(sum/size) + ","

text_file.write(s + '\n')

text_file.close()