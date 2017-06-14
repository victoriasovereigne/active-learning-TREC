import os, pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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

def calculate_precision_recall_f1(train_index_list, X, y, 
    learning_curve, learning_batch_size, batch_size):
    print "Welcome to helper PRCF1"
    y_pred_all = {}

    for train_index in train_index_list:
        y_pred_all[train_index] = y[train_index]

    for train_index in xrange(0, len(X)):
        if train_index not in train_index_list:
            y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

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

    return precision, recall, f1score