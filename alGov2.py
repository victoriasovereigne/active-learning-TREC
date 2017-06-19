'''Active learning for labeling the relevant document for TREC-8 dataset
@author: Md Mustafizur Rahman (nahid@utexas.edu)'''

import os
import numpy as np
import sys
sys.setrecursionlimit(100000000)
from bs4 import BeautifulSoup
import re
import nltk
import copy
import bz2


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

import pickle

import logging
logging.basicConfig()

np.random.seed(1335)
TEXT_DATA_DIR = '/home/nahid/TREC/gov2/'
RELEVANCE_DATA_DIR = '/home/nahid/TREC/qrels.tb06.top50.txt'
topic_number = '410'
docrepresentation = "TF-IDF"  # can be BOW, TF-IDF
sampling=True # can be True or False
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

all_reviews = {}

if preloaded==False:
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        print name
        #if name == "ft":
        path = os.path.join(TEXT_DATA_DIR, name)
        print path
        if os.path.isfile(path):
            #f = open(path)
            bz_file = bz2.BZ2File(path)

            s = bz_file.readlines()
            #str1 = ' '.join(s)
            #s = ""
            length = len(s)
            tmp=""
            accumulate = False

            for i in xrange(0,length):
                s[i] = str(s[i])
                #s[i] = str(s[i]).replace(" ", "").replace("\t", "")
                '''print s[i]
                for j in xrange(0,len(s[i])):
                    print j,":", s[i][j]
                print len(s[i])
                print type(s[i])
                print type("<DOC>")
                print len("<DOC>")
                '''
                if s[i].find("<DOC>")!=-1:
                    #print "Doc Start"
                    tmp=tmp+s[i]
                    accumulate = True

                if accumulate == True:
                    tmp=tmp+s[i]

                if s[i].find("</DOC>")!=-1:
                    tmp=tmp+s[i]
                    accumulate = False
                    #print "Doc End"
                    #print tmp
                    soup = BeautifulSoup(tmp, 'html.parser')
                    docs = soup.find_all('doc')
                    #print docs[0]
                    tmp = ""
                    soup = BeautifulSoup(str(docs[0]), 'html.parser')
                    docNo = soup.find_all('docno')[0].next.replace(" ", "").replace("\t", "")
                    #p = soup.find_all("p")
                    print docNo
                    #print len(p)
                    #text=soup.get_text()

                    #print p[0].next
                    #print p[len(p)-1].next
                    #print docsNos[0]
                    #print len(docsNos)
                    #print text
                    '''
                    if len(docsNos) != len(headLines):
                        print "UNEQUAL"
                        print len(docsNos)
                        print len(headLines)
                        print len(texts)
                    '''

                    if docNo in docNo_label:
                        all_reviews[docNo] = review_to_words(soup.get_text())
                        print "in List", docNo



            f.close()

    output = open('/home/nahid/TREC/govcombine/output.txt', 'ab+')
    # data = {'a': [1, 2, 3], }

    pickle.dump(all_reviews, output)
    output.close()

else:
    input = open('/home/nahid/TREC/data/output.txt', 'rb')
    all_reviews = pickle.load(input)
    print "pickle loaded"

s = "";
#for topic in sorted(topic_to_doclist.keys()):
for topic in xrange(801,850):
    print "Topic:", topic
    topic = str(topic)
    docList = topic_to_doclist[topic]
    print docList
    print ('Processing news text for topic number')
    relevance_label = []
    judged_review = []

    for documentNo in docList:
        if all_reviews.has_key(documentNo):
            print "in List", documentNo
            judged_review.append(all_reviews[documentNo])
            relevance_label.append(docNo_label[documentNo])

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

    datasize = len(X)
    print "Whole Dataset size: ", datasize

    #print len(y)
    #print y
    numberOne = y.count(1)
    print "Number of One", numberOne

    numberZero = y.count(0)
    print "Number of zero", numberZero


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

    n_labeled = int(len(y_train)*0.98)

    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query
    print "quotas:", quota

    batch_size = int(quota / 10)
    #quota = 50

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()

    E_in_1, E_out_1, model = run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1score = f1_score(y_test, y_pred, average='binary')

    print "precision score:", precision
    print "recall score:", recall
    print "f-1 score:", f1score


    s=s+topic+","+str(datasize)+","+str(numberOne)+","+str(numberZero)+","+str(precision)+","+str(recall)+","+str(f1score)+"\n";

    text_file = open("/home/nahid/TREC/results_analysis_450.txt", "w")
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


