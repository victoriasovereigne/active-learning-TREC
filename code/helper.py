import os, pickle, re
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import copy
from nltk.corpus import stopwords

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsemble

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

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


def get_ranker(ranker_location, datasource):
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
    return Ranker_topic_to_doclist

def load_pickle(TEXT_DATA_DIR, processed_file_location):
    files = os.listdir(TEXT_DATA_DIR)
    all_reviews = {}

    for name in sorted(files):
        path = os.path.join(TEXT_DATA_DIR, name)
        print path
        f = open(path)
        docNo = name[0:name.index('.txt')]

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
        for lines in f:
            if c < count:
                c = c + 1
                continue
            values = lines.split()
            c = c + 1
            tmpStr = tmpStr + " "+ str(values[2])
        print tmpStr
        #exit(0)

        #if docNo in docNo_label:
        all_reviews[docNo] = (review_to_words(tmpStr))
        f.close()

    output = open(processed_file_location, 'ab+')
    pickle.dump(all_reviews, output)
    output.close()
    return all_reviews

# f = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dv_auto_asr'
# processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/clef2007/processed_auto_asr.txt'
# load_pickle(f, processed_file_location)

# f = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dv_auto_both'
# processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/clef2007/processed_auto_both.txt'
# load_pickle(f, processed_file_location)

# f = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dv_auto_keyword'
# processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/clef2007/processed_auto_keyword.txt'
# load_pickle(f, processed_file_location)

# f = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dv_manual_keyword'
# processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/clef2007/processed_manual_keyword.txt'
# load_pickle(f, processed_file_location)

# f = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dv_manual_summary'
# processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/clef2007/processed_manual_summary.txt'
# load_pickle(f, processed_file_location)

# f = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/dv_manual_both'
# processed_file_location = '/v/filer4b/v20q001/vlestari/Documents/Summer/IR/clef2007/processed_manual_both.txt'
# load_pickle(f, processed_file_location)

def rank_topic_to_doclist(ranker_location, datasource):
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
	return Ranker_topic_to_doclist


def create_initial_training_set(X, y, counter, limit, train_index_list, initial_X_train, initial_y_train, clas):
    for index in xrange(0,len(X)):
        if y[index] == clas:
            counter = counter + 1
            train_index_list.append(index)
            initial_X_train.append(X[index])
            initial_y_train.append(y[index])
            print counter
        if counter == limit :
            break

    return initial_X_train, initial_y_train

def fit_model(model, ens, protocol, correction, under_sampling, 
                initial_X_train, initial_y_train, sampling_weight,
                initial_X_agg, initial_y_agg):
    # if protocol == 'SPL':
        # model = LogisticRegression()

    if correction == True:
        model.fit(initial_X_train, initial_y_train, sample_weight=sampling_weight)
    else:
        if under_sampling == True:
            ens.fit(initial_X_agg, initial_y_agg)
            print("ensemble works")
        else:
            model.fit(initial_X_train, initial_y_train)


def get_prediction_y_pred(train_index_list, docIndex_DocNo, topic, human_label_location, 
                            train_per_centage, loopCounter, under_sampling, ens, model, X, y):
    y_pred_all = {}
    human_label_location_final, human_label_str = write_human_label_location(train_index_list, 
                                    y, y_pred_all, docIndex_DocNo, topic, human_label_location, 
                                    train_per_centage[loopCounter])

    for train_index in xrange(0, len(X)):
        if train_index not in train_index_list:
            if under_sampling == True:
                y_pred_all[train_index] = ens.predict(np.array(X[train_index]).reshape(1, -1))
            else:
                y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

    return y_pred_all

def write_pred_to_file(y_pred_all, X, docIndex_DocNo, topic, predicted_location_base, train_per_centage, loopCounter):
    y_pred = []
    for key, value in y_pred_all.iteritems():
        y_pred.append(value)
    
    pred_topic_str = ""
    for docIndex in xrange(0, len(X)):
        docNo = docIndex_DocNo[docIndex]
        pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"
    
    predicted_location_final = predicted_location_base + str(train_per_centage[loopCounter]) + '.txt'
    text_file = open(predicted_location_final, "a")
    text_file.write(pred_topic_str)
    text_file.close()

    return y_pred

def get_f1score(y, y_pred, learning_curve, index):
    f1score = f1_score(y, y_pred, average='binary')

    if (learning_curve.has_key(index)):
        tmplist = learning_curve.get(index)
        tmplist.append(f1score)
        learning_curve[index] = tmplist
    else:
        tmplist = []
        tmplist.append(f1score)
        learning_curve[index] = tmplist

    return f1score

def get_precision(y, y_pred, learning_curve, index):
    precision = precision_score(y, y_pred, average='binary')

    if (learning_curve.has_key(index)):
        tmplist = learning_curve.get(index)
        tmplist.append(precision)
        learning_curve[index] = tmplist
    else:
        tmplist = []
        tmplist.append(precision)
        learning_curve[index] = tmplist

    return precision

def get_recall(y, y_pred, learning_curve, index):
    recall = recall_score(y, y_pred, average='binary')

    if (learning_curve.has_key(index)):
        tmplist = learning_curve.get(index)
        tmplist.append(recall)
        learning_curve[index] = tmplist
    else:
        tmplist = []
        tmplist.append(recall)
        learning_curve[index] = tmplist

    return recall

def get_sensitivity_specificity(y, y_pred, learning_curve, index):
    C = confusion_matrix(y, y_pred)
    print "---------------------------"
    print "Confusion matrix"
    print "---------------------------"
    print C
    print "---------------------------"

    TN = C[0][0]
    FP = C[0][1]
    FN = C[1][0]
    TP = C[1][1]

    sensitivity = float(TP) / (TP + FN)
    specificity = float(TN) / (TN + FP)

    if (learning_curve.has_key(index)):
        tmplist = learning_curve.get(index)
        tmplist.append(specificity)
        learning_curve[index] = tmplist
    else:
        tmplist = []
        tmplist.append(specificity)
        learning_curve[index] = tmplist

    return sensitivity, specificity    


def update_initial_train(iter_sampling, under_sampling, smote, unmodified_train_X, unmodified_train_y, num_subsets):
    if iter_sampling == True:
        print "Oversampling in the active iteration list"
        ros = RandomOverSampler()
        initial_X_train = None
        initial_y_train = None
        initial_X_train, initial_y_train = ros.fit_sample(unmodified_train_X, unmodified_train_y)
    elif under_sampling == True:
        ee = EasyEnsemble(return_indices=True, replacement=True, n_subsets=num_subsets)
        initial_X_train = None
        initial_y_train = None
        initial_X_train, initial_y_train, indices = ee.fit_sample(unmodified_train_X, unmodified_train_y)
    elif smote == True:
        ros = SMOTE(k_neighbors=3)
        initial_X_train = None
        initial_y_train = None
        initial_X_train, initial_y_train = ros.fit_sample(unmodified_train_X, unmodified_train_y)
    else:
        # initial_X_train[:] = []
        # initial_y_train[:] = []
        initial_X_train = copy.deepcopy(unmodified_train_X)
        initial_y_train = copy.deepcopy(unmodified_train_y)

    return initial_X_train, initial_y_train

def write_predicted_location(X, docIndex_DocNo, topic, y_pred_all, 
    predicted_location_base, index):
    pred_topic_str = ""
    for docIndex in xrange(0, len(X)):
        docNo = docIndex_DocNo[docIndex]
        pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"

    predicted_location_final = predicted_location_base + str(index) + '.txt'
    text_file = open(predicted_location_final, "a")
    text_file.write(pred_topic_str)
    text_file.close()

def write_human_label_location(train_index_list, y, y_pred_all, docIndex_DocNo, topic, 
    human_label_location, index):
    human_label_str = ""

    for train_index in train_index_list:
        y_pred_all[train_index] = y[train_index]
        docNo = docIndex_DocNo[train_index]
        human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[train_index]) + "\n"
    
    human_label_location_final = human_label_location + str(index) + '_human_.txt'
    text_file = open(human_label_location_final, "a")
    text_file.write(human_label_str)
    text_file.close()

    return human_label_location_final, human_label_str

def create_test_index_list(X, y, train_index_list, initial_X_test, initial_y_test):
    test_index_list = {}
    test_index_counter = 0
    for train_index in xrange(0, len(X)):
        if train_index not in train_index_list:
            initial_X_test.append(X[train_index])
            test_index_list[test_index_counter] = train_index
            test_index_counter = test_index_counter + 1
            initial_y_test.append(y[train_index])

    return test_index_list

def predict_y_pred_all(X, train_index_list, y_pred_all, model, ens, under_sampling):
    print "update_y_pred_and_all"
    print "================================"

    for train_index in xrange(0, len(X)):
        if train_index not in train_index_list:
            if under_sampling:
                print "under_sampling"
                y_pred_all[train_index] = ens.predict(np.array(X[train_index]).reshape(1, -1))
                print y_pred_all[train_index]
            else:
                y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

    return y_pred_all

def fill_agg(initial_X_train, initial_y_train, num_subsets):
    initial_X_agg = []
    initial_y_agg = []

    for i in xrange(0, num_subsets):
        tmplist = copy.deepcopy(initial_X_train)
        initial_X_agg.append(tmplist)

        tmplist = copy.deepcopy(initial_y_train)
        initial_y_agg.append(tmplist)

    # print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"
    # print np.array(initial_X_train).shape, np.array(initial_y_train).shape
    # print np.array(initial_X_agg).shape, np.array(initial_y_agg).shape
    # print "@#$%^&*(*&^%$#$%^&*(*&^%$#@#$%^&*(*&^%$#@#$%^&**&^%$#@#$%^&*"

    return np.array(initial_X_agg), np.array(initial_y_agg)