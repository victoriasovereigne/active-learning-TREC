import os
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
import nltk
import copy

#nltk.download()  # Download text data sets, including stop words

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn import svm
import matplotlib.pyplot as plt
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler


from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from imblearn.over_sampling import RandomOverSampler

import logging
logging.basicConfig()

np.random.seed(1335)



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
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out

BASE_DIR = ''
GLOVE_DIR = '/home/nahid/PycharmProjects/kerasLearning' + '/glove.6B/'
TEXT_DATA_DIR = '/home/nahid/TREC/v4/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2






print ('Processing news text')
#collect all the news as a text in one index of a list
texts = []
#corresponding label of that news in a list
label = []
# we have in total 20 news label, for each news label we have a directory so we will store the label from there as dictionary
# label_name --> label_index
docno_rawtext = {} #original text with headline appended at the first
docno_bagtext = {}  # bag of word models
docno_docindex = {}
clear_review = []

for name in sorted(os.listdir(TEXT_DATA_DIR)):
    print name
    if name == "ft":
        path = os.path.join(TEXT_DATA_DIR, name)
        print path
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                print fname
                fpath = os.path.join(path,fname)
                print fpath
                if os.path.isdir(fpath):
                    for fpname in sorted(os.listdir(fpath)):
                        fpname = os.path.join(fpath,fpname)
                        print fpname

                        f = open(fpname)
                        #texts.append(f.read()) # appending all the news from f as a text
                        s = f.read()
                        #print s

                        soup = BeautifulSoup(s, 'html.parser')
                        #print soup.find_all('docno')[0].next
                        docsNos = soup.find_all('docno')
                        headLines = soup.find_all('headline')
                        texts = soup.find_all('text')
                        #print len(docsNos)

                        #print len(headLines)
                        #print len(texts)

                        if len(docsNos) != len(headLines):
                            print "UNEQUAL"
                            print len(docsNos)
                            print len(headLines)
                            print len(texts)
                        for i in xrange(0, min(len(docsNos),len(headLines), len(texts))):
                            #print docsNos[i].next
                            #print texts[i].next
                            #print docsNos[i].next
                            docno_rawtext[docsNos[i].next] = texts[i].next
                            docNumber = len(docno_docindex)
                            docno_docindex[docsNos[i].next] = docNumber
                            clear_review.append(review_to_words(docno_rawtext[docsNos[i].next]))
                            #print docno_text[docsNos[i].next]
                        f.close()


#print len(docno_docindex)
#for s in docno_docindex:
#    print s
relevance_label = [0]*len(docno_docindex)
print len(relevance_label)
print('Reading the relevance label')
# file open
f = open('/home/nahid/relevance.txt')
print f
for lines in f:
    values = lines.split()
    # print values[0]
    if values[0] == '401':
        label = int(values[3])

        docNo = values[2]
        #print docNo, label
        if docNo[0]=='F':
            if docNo[1]=='T':
                if docno_docindex.has_key(docNo):
                    #print "has key"
                    docIndex = docno_docindex[docNo]
                    #print docIndex
                    relevance_label[docIndex] = label
f.close()
#print relevance_label

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
bag_of_word = vectorizer.fit_transform(clear_review)

# Numpy arrays are easy to work with, so convert the result to an
# array
bag_of_word = bag_of_word.toarray()
#print bag_of_word.shape
#vocab = vectorizer.get_feature_names()
#print vocab

# Sum up the counts of each vocabulary word
#dist = np.sum(bag_of_word, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
#for tag, count in zip(vocab, dist):
#    print count, tag

print "Bag of word completed"

X=[]
y=[]

f = open('/home/nahid/relevance.txt')
print f
for lines in f:
    values = lines.split()
    # print values[0]
    if values[0] == '401':
        label = int(values[3])

        docNo = values[2]
        #print docNo, label
        if docNo[0]=='F':
            if docNo[1]=='T':
                if docno_docindex.has_key(docNo):
                    #print "has key"
                    docIndex = docno_docindex[docNo]
                    #print docIndex
                    X.append(bag_of_word[docIndex])
                    y.append(relevance_label[docIndex])
f.close()

print len(X)
print len(y)
#print y
print y.count(1)
print y.count(0)


ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(X, y)


print 'Resampled Version'
print len(X_resampled)
print len(y_resampled)
#print y
print y_resampled.tolist().count(1)
print y_resampled.tolist().count(0)


#X = X_resampled
#y = y_resampled

test_size = 0.6    # the percentage of samples in the dataset that will be
n_labeled = 10      # number of samples that are initially labeled

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size)
trn_ds = Dataset(X_train, np.concatenate(
    [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
tst_ds = Dataset(X_test, y_test)
fully_labeled_trn_ds = Dataset(X_train, y_train)

trn_ds2 = copy.deepcopy(trn_ds)
lbr = IdealLabeler(fully_labeled_trn_ds)

quota = len(y_train) - n_labeled    # number of samples to query
print "quotas:", quota

# Comparing UncertaintySampling strategy with RandomSampling.
# model is the base learner, e.g. LogisticRegression, SVM ... etc.
qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
model = LogisticRegression()
E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)


# Plot the learning curve of UncertaintySampling to RandomSampling
# The x-axis is the number of queries, and the y-axis is the corresponding
# error rate.
query_num = np.arange(1, quota + 1)
#plt.plot(query_num, E_in_1, 'g', label='qs Ein')
#plt.plot(query_num, E_in_2, 'r', label='random Ein')
plt.plot(query_num, E_out_1, 'b', label='')
plt.xlabel('Number of Queries')
plt.ylabel('Error')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
plt.show()

