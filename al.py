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

import logging
logging.basicConfig()

np.random.seed(1335)
TEXT_DATA_DIR = '/home/nahid/TREC/data/'
RELEVANCE_DATA_DIR = '/home/nahid/TREC/relevance.txt'
topic_number = '404'
docrepresentation = "BOW"  # can be BOW, TF-IDF
sampling=True # can be True or False
test_size = 0.5    # the percentage of samples in the dataset that will be
n_labeled = 10      # number of samples that are initially labeled




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


def run(trn_ds, tst_ds, lbr, model, qs, quota):
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

    return E_in, E_out




topic_to_doclist = {} # key is the topic(string) and value is the list of docNumber
docNo_label = {} # key is the DocNo and the value is the label
print('Reading the relevance label')
# file open
f = open(RELEVANCE_DATA_DIR)
print f
tmplist = []
for lines in f:
    values = lines.split()
    topic = values[0]
    docNo = values[2]
    label = int(values[3])
    docNo_label[docNo] = label
    if(topic_to_doclist.has_key(topic)):
        tmplist.append(docNo)
    else:
        tmplist = []
        tmplist.append(docNo)
        topic_to_doclist[topic] = tmplist
f.close()


#for topic in sorted(topic_to_doclist.keys()):
for topic in ["401"]:
    print "Topic:", topic
    docList = topic_to_doclist[topic]
    print docList
    print ('Processing news text for topic number')

    relevance_label = []
    judged_review = []

    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        print name
        #if name == "ft":
        path = os.path.join(TEXT_DATA_DIR, name)
        print path
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                #print fname
                fpath = os.path.join(path,fname)
                print fpath
                if os.path.isdir(fpath):
                    for fpname in sorted(os.listdir(fpath)):
                        fpname = os.path.join(fpath,fpname)
                        #print fpname

                        f = open(fpname)
                        s = f.read()

                        soup = BeautifulSoup(s, 'html.parser')
                        docsNos = soup.find_all('docno')
                        texts = soup.find_all('text')
                        '''
                        if len(docsNos) != len(headLines):
                            print "UNEQUAL"
                            print len(docsNos)
                            print len(headLines)
                            print len(texts)
                        '''
                        for i in xrange(0, min(len(docsNos), len(texts))):
                            #print docsNos[i].next
                            docNo = docsNos[i].next.replace(" ", "").replace("\t", "")
                            if docNo in docList:
                                print "in List", docsNos[i].next

                                judged_review.append(review_to_words(texts[i].next))
                                relevance_label.append(docNo_label[docNo])
                        f.close()
                else:
                    f = open(fpath)
                    s = f.read()

                    soup = BeautifulSoup(s, 'html.parser')
                    docsNos = soup.find_all('docno')
                    texts = soup.find_all('text')
                    for i in xrange(0, min(len(docsNos), len(texts))):
                        # print docsNos[i].next
                        docNo = docsNos[i].next.replace(" ", "").replace("\t", "")
                        if docNo in docList:
                            print "in List", docsNos[i].next

                            judged_review.append(review_to_words(texts[i].next))
                            relevance_label.append(docNo_label[docNo])
                    f.close()



    if docrepresentation == "TF-IDF":
        print "Using TF-IDF"
        vectorizer = TfidfVectorizer(min_df=5, \
                                 analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)

        bag_of_word = vectorizer.fit_transform(judged_review)


    elif docrepresentation == "BOW":
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        print "Uisng Bag of Word"
        vectorizer = CountVectorizer(analyzer = "word",   \
                                     tokenizer = None,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     max_features = 5000)

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


    print "Whole Dataset size: ", len(X)
    #print len(y)
    #print y
    print "Number of One", y.count(1)
    print "Number of zero", y.count(0)

    #This stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.
    #For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, stratify=y)

    if sampling == True:
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_sample(X_train, y_train)
        X_train = X_train.tolist()
        y_train = y_train.tolist()
        print "Before", y_train
        print "Number of one in train after sampling", y_train.count(1)
        print "Number of one in test after sampling", y_test.count(1)

        # we have to do this because randomoversampling placing all the zero at the first halh
        # and all the one label at last half
        # which is creating problem for activer learning (logistic regression module)
        # we are passing the first 10 sample and becuase of this the first ten sample
        # only contains zero

        X_a, X_b, y_a, y_b = \
            train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)

        X_train = X_a + X_b
        y_train = y_a + y_b

        print "After", y_train

    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query
    print "quotas:", quota
    if quota>30:
        quota = 30
    #quota = 10

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()

    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)
    y_pred = model.predict(X_test)

    print "f-1 score:", f1_score(y_test, y_pred, average=None)

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    #plt.plot(query_num, E_in_1, 'g', label='qs Ein')
    #plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'b', label='')
    print E_out_1
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


