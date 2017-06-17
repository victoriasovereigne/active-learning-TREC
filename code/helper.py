import os, pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import copy

from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import EasyEnsemble

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
	all_reviews = {}

	for name in sorted(os.listdir(TEXT_DATA_DIR)):
		path = os.path.join(TEXT_DATA_DIR, name)
        print path

        f = open(path)
        docNo = name[0:name.index('.')]

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
    if protocol == 'SPL':
        model = LogisticRegression()

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
    # human_label_str = ""

    human_label_location_final, human_label_str = write_human_label_location(train_index_list, 
                                    y, y_pred_all, docIndex_DocNo, topic, human_label_location, 
                                    train_per_centage[loopCounter])
    # for train_index in train_index_list:
    #     y_pred_all[train_index] = y[train_index]
    #     docNo = docIndex_DocNo[train_index]
    #     human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[train_index]) + "\n"
        
    # human_label_location_final = human_label_location + str(train_per_centage[loopCounter]) + '_human_.txt'
    # text_file = open(human_label_location_final, "a")
    # text_file.write(human_label_str)
    # text_file.close()

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

def get_prec_recall_f1(y, y_pred, learning_curve, train_per_centage, loopCounter,
    learning_batch_size, batch_size):
    f1score = f1_score(y, y_pred, average='binary')

    if (learning_curve.has_key(train_per_centage[loopCounter])):
        tmplist = learning_curve.get(train_per_centage[loopCounter])
        tmplist.append(f1score)
        learning_curve[train_per_centage[loopCounter]] = tmplist
    else:
        tmplist = []
        tmplist.append(f1score)
        learning_curve[train_per_centage[loopCounter]] = tmplist

    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')

    return precision, recall, f1score

def update_initial_train(iter_sampling, under_sampling, unmodified_train_X, unmodified_train_y, num_subsets):
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
    else:
        initial_X_train[:] = []
        initial_y_train[:] = []
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
    # print "write_human_label_location"
    # print train_index_list
    # print "================================"

    for train_index in train_index_list:
        y_pred_all[train_index] = y[train_index]
        docNo = docIndex_DocNo[train_index]
        human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[train_index]) + "\n"
    
    human_label_location_final = human_label_location + str(index) + '_human_.txt'
    text_file = open(human_label_location_final, "a")
    text_file.write(human_label_str)
    text_file.close()

    # print "================================"
    # print y_pred_all

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