import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.preprocessing import *
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


data_dir = '/home/nahid/Downloads/data/user-resource.csv'  # needs trailing slash


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariateGaussian(dataset, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)


def selectThresholdByCV(probs, gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs), max(probs), stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(gt, predictions, average='binary')
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon

    return best_f1, best_epsilon

train_file = data_dir
df = pd.read_csv(train_file)
print df.dtypes
#print df.shape
#print df.groupby(['username', 'ip', 'accesstype', 'repository']).agg({'repository': np.size})

accesstypeToIndex = {}
df_accesstype = pd.get_dummies(df['accesstype'])
#print df_accesstype.dtypes
# accessType to accesIndexMapper
accessTypeList = []
index = 0
for i, row in df_accesstype.dtypes.iteritems():
    print index, i
    accesstypeToIndex[i] = index
    index = index + 1
    accessTypeList.append(i)


#print len(accesstypeToIndex)
numberOfAccessType = len(accesstypeToIndex)
user_to_activities = {} # key is the userId and value is the list of activities, number of ip address
user_to_ip = {} # key is the userID and value is the list of unique ip address
index_to_userid = {}
userid_to_index = {}
userCounter = 0
unidentifiableUserCounter = 0
sshUser =''
specialEventUser = ''
for index, row in df.iterrows():
    userid = row["username"]

    if userid == "-":
        unidentifiableUserCounter = unidentifiableUserCounter + 1
        continue
    #print userid

    ipaddress = row["ip"]
    accessTypeId = accesstypeToIndex[row["accesstype"]]

    if row["accesstype"] == "SshAccessKeyGrantedEvent":
        sshUser = userid

    if row["accesstype"] == "PullRequestConditionDeletedEvent":
        specialEventUser = userid

    if(user_to_activities.has_key(userid)):
        tmplist = user_to_activities[userid]
        tmplist[accessTypeId] = tmplist[accessTypeId] + 1
        user_to_activities[userid] = tmplist
    else:
        index_to_userid[userCounter] = userid
        userid_to_index[userid] = userCounter
        userCounter = userCounter + 1
        tmplist = [0] * numberOfAccessType
        tmplist[accessTypeId] = tmplist[accessTypeId] + 1
        user_to_activities[userid] = tmplist

    if (user_to_ip.has_key(userid)):
        ipaddressList = user_to_ip[userid]
        if ipaddress not in ipaddressList:
            ipaddressList.append(ipaddress)
        user_to_ip[userid] = ipaddressList
    else:
        ipaddressList = []
        ipaddressList.append(ipaddress)
        user_to_ip[userid] = ipaddressList

print "Number of users", len(user_to_activities)
print "Number of unidentifiable users:", unidentifiableUserCounter
#print user_to_activities["-"]
print "PullRequestConditionDeletedEvent user:", specialEventUser
print "SshAccessKeyGrantedEvent user:", sshUser


featureMatrix = []
normalizedFeatureMatrix = []
for user, features in user_to_activities.iteritems():
    #print user, features , len(features)
    #print len(user_to_ip[user])
    # appending the number of unique ipaddress used by an user in the final features list
    #features.append(len(user_to_ip[user]))
    featureMatrix.append(features)
    #print user, features, len(features)


print featureMatrix[0], len(featureMatrix)
print "Accumulated accessTypeEvent", np.sum(featureMatrix, axis=0)




# nomalizing the column
'''featuredf = pd.DataFrame(featureMatrix)
for col in xrange(0, numberOfAccessType+1):
    print col
    max_value = featuredf[col].max()
    min_value = featuredf[col].min()
    featuredf[col] = (featuredf[col] - min_value) / (max_value - min_value)
'''
x = np.array(featureMatrix)
#normalizedFeatureMatrix = x / (x.sum(0) if x.sum(0)>0 else 1)

min_max_scaler = MinMaxScaler()
normalizedFeatureMatrix = min_max_scaler.fit_transform(x)

#print x
#print df.groupby(['username', 'accesstype']).agg({'accesstype': np.size})
#print df.groupby(['username', 'ip', 'repository']).agg({'ip': np.size})
#print df.groupby('repository')['username'].count()
#print df[df['username'] == 'a35fe80c797b84cfff4c4c94441d90aa48472144c5603995e96849dd']



'''
for i, row in df_accesstype.dtypes.iteritems():
    print i, accesstypeToIndex[i]
'''
#print type(df_accesstype.dtypes)
#print df_accesstype.shape
'''df_repository = pd.get_dummies(df['repository'])
df_resource = pd.get_dummies(df['resource'])
df_project = pd.get_dummies(df['project'])
df_new = pd.concat([df_repository, df_accesstype, df_resource, df_project], axis=1)
'''
#print df_new.shape
#print df_new.dtypes
#print df_new.iloc[[2]]
#df_newest = df_new.drop(['username'], axis=1)
#print df_newest.dtypes
'''df = None


# Convert DataFrame to matrix
mat = df_new.as_matrix()
df_new = None
'''

objects = accessTypeList
y_pos = np.arange(numberOfAccessType)
accumulatedAccesses = np.sum(featureMatrix, axis=0)

plt.barh(y_pos, accumulatedAccesses, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accumulated AccessType Events')
plt.title('Summarization of AccessType Events')
plt.show()



from scipy.spatial.distance import cdist
clusters=range(2,15)
meandist=[]

for k in clusters:
    model=KMeans(init='k-means++', n_clusters=k)
    model.fit(normalizedFeatureMatrix)
    clusassign=model.predict(normalizedFeatureMatrix)
    meandist.append(sum(np.min(cdist(normalizedFeatureMatrix, model.cluster_centers_, 'euclidean'), axis=1)) / normalizedFeatureMatrix.shape[0])

#Plot average distance from observations from the cluster centroid
#to use the Elbow Method to identify number of clusters to choose
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.show()


for n_cluster in range(2, 11):
    kmeans = KMeans(init='k-means++', n_clusters=n_cluster).fit(normalizedFeatureMatrix)
    label = kmeans.labels_
    sil_coeff = silhouette_score(normalizedFeatureMatrix, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))


# Using sklearn
km = KMeans(init='k-means++',n_clusters=10)
km.fit(normalizedFeatureMatrix)
# Get cluster assignment labels
labels = km.labels_
print km.cluster_centers_
print labels, len(labels)

counter = 0
k_meansList = []
labelList = {}
for a in xrange(0, len(labels)):
    '''if labels[a] == 1:
        print index_to_userid[a], featureMatrix[a]
        counter = counter + 1
        k_meansList.append(index_to_userid[a])
    '''
    if labelList.has_key(labels[a]):
        tmpLabelList = labelList.get(labels[a])
        tmpLabelList.append(index_to_userid[a])
        labelList[labels[a]] = tmpLabelList
    else:
        tmpLabelList = []
        tmpLabelList.append(index_to_userid[a])
        labelList[labels[a]] = tmpLabelList


perCategoryUserList = []
for label, userList in labelList.iteritems():
    print "##########Label %d ###########" %(label)
    print "Number of users in this catergory:", len(userList)
    perCategoryUserList.append(len(userList))
    for user in userList:
        print user, featureMatrix[userid_to_index[user]]

cateGoryNumber = 1
for numberofUserperCategory in perCategoryUserList:
    print "Category %d: Number of Users:%d" %(cateGoryNumber, numberofUserperCategory)
    cateGoryNumber = cateGoryNumber + 1
print "k_means is Done", counter


''' # Analysing category with few member
import operator
mylist = [13, 0, 0, 0, 10, 2, 0, 11, 21, 51, 51, 0, 0, 0, 0, 0, 63, 0, 932, 17, 0, 2, 2, 1, 1, 1, 25, 7, 192, 54, 17, 17, 13, 13, 3, 3, 99, 7, 7, 0]
index, value = max(enumerate(mylist), key=operator.itemgetter(1))
print index, value
18 932
'''




'''
#Gaussian Analysis
mu, sigma = estimateGaussian(x)
print mu, sigma
p = multivariateGaussian(x,mu,sigma)
ep = 9.036201327981216e-05
outliers = np.asarray(np.where(p < ep))
print "#####Gaussian####", outliers
'''


#Oneclass SVM
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf.fit(x)
pred = clf.predict(x)
print len(pred)

# inliers are labeled 1, outliers are labeled -1
normal = x[pred == 1]
abnormal = x[pred == -1]

unusual = 0
for iter in xrange(0, len(x)):
    if pred[iter]< 0:
        unusual = unusual + 1
        print index_to_userid[iter], featureMatrix[iter]

print "Unusual user accoring to OneClassSVC:",unusual


'''
overLapped = 0
overLappedList =  []
for iter in xrange(0, len(x)):
    if pred[iter]< 0:
        print index_to_userid[iter], featureMatrix[iter]
        if index_to_userid[iter] in k_meansList:
            overLapped = overLapped + 1
            overLappedList.append(index_to_userid[iter])


print "OverLapped User:", len(overLappedList)
for user in overLappedList:
    print user, featureMatrix[userid_to_index[user]]

'''


rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(x)
y_pred= clf.predict(x)
unusual = 0
for iter in xrange(0, len(x)):
    if y_pred[iter]< 0:
        unusual = unusual + 1
        print index_to_userid[iter], featureMatrix[iter]

print "Unusual user accoring to Random State:",unusual


# Format results as a DataFrame
#results = pd.DataFrame([df.index,labels]).T
#print len(results)
#results = pd.DataFrame(data=labels, columns=['cluster'], index=collapsed.index)

'''
v = DictVectorizer()
qualitative_features = ['repository','accesstype','resource', 'project']
X_qual = v.fit_transform(df[qualitative_features].to_dict('records'))
print v.vocabulary_
print X_qual.toarray()[0]
'''

'''
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df'''


'''
df['repository'] = df['repository'].astype('category')


print df.dtypes
#print df.repository
df.repository.cat.categories
print df.repository.cat.codes
'''
'''
x = pd.get_dummies(train)

print x.shape
print type(x)
print x
'''
#test = pd.read_csv(test_file)

# set missing YOB to zero
'''
train.YOB[train.YOB.isnull()] = 0
train.YOB[train.YOB < 1920] = 0
train.YOB[train.YOB > 2004] = 0


test.YOB[test.YOB.isnull()] = 0
test.YOB[test.YOB < 1920] = 0
test.YOB[test.YOB > 2004] = 0

# numeric x

numeric_cols = ['YOB', 'votes']
x_num_train = train[numeric_cols].as_matrix()
x_num_test = test[numeric_cols].as_matrix()

# scale to <0,1>

max_train = np.amax(x_num_train, 0)
max_test = np.amax(x_num_test, 0)  # not really needed

x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train  # scale test by max_train

# y

y_train = train.Happy
y_test = test.Happy
'''
# categorical

'''cat_train = train

#cat_train = train.drop(['username'], axis=1)
#cat_test = test.drop(numeric_cols + ['UserID', 'Happy'], axis=1)

cat_train.fillna('NA', inplace=True)
#cat_test.fillna('NA', inplace=True)

x_cat_train = cat_train.to_dict(orient='records')
#x_cat_test = cat_test.to_dict(orient='records')

print x_cat_train[0]
'''
'''
# vectorize
vectorizer = DV(sparse=False)
vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
#vec_x_cat_test = vectorizer.transform(x_cat_test)

# complete x

x_train = np.hstack((vec_x_cat_train))
#x_train = np.hstack((x_num_train, vec_x_cat_train))
#x_test = np.hstack((x_num_test, vec_x_cat_test))

if __name__ == "__main__":
    # SVM looks much better in validation

    print "training SVM..."

    # although one needs to choose these hyperparams
    C = 173
    gamma = 1.31e-5
    shrinking = True

    probability = True
    verbose = True

    svc = SVC(C=C, gamma=gamma, shrinking=shrinking, probability=probability, verbose=verbose)
    svc.fit(x_train, y_train)
    p = svc.predict_proba(x_test)

    auc = AUC(y_test, p[:, 1])
    print "SVM AUC", auc

    print "training random forest..."

    n_trees = 100
    max_features = int(round(sqrt(x_train.shape[1]) * 2))  # try more features at each split
    max_features = 'auto'
    verbose = 1
    n_jobs = 1

    rf = RF(n_estimators=n_trees, max_features=max_features, verbose=verbose, n_jobs=n_jobs)
    rf.fit(x_train, y_train)

    p = rf.predict_proba(x_test)

    auc = AUC(y_test, p[:, 1])
    print "RF AUC", auc

    # AUC 0.701579086548
    # AUC 0.676126704696

    # max_features * 2
    # AUC 0.710060065732
    # AUC 0.706282346719
'''