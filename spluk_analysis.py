'''Code developed by Md Mustafizur Rahman
E-mail: nahid@utexas.edu
'''

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.preprocessing import *
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


data_dir = '/home/nahid/Downloads/data/user-resource.csv'

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

accesstypeToIndex = {}
df_accesstype = pd.get_dummies(df['accesstype'])

accessTypeList = []
index = 0
for i, row in df_accesstype.dtypes.iteritems():
    print index, i
    accesstypeToIndex[i] = index
    index = index + 1
    accessTypeList.append(i)

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

print "PullRequestConditionDeletedEvent user:", specialEventUser
print "SshAccessKeyGrantedEvent user:", sshUser


featureMatrix = []
normalizedFeatureMatrix = []
for user, features in user_to_activities.iteritems():
    featureMatrix.append(features)


print featureMatrix[0], len(featureMatrix)
print "Accumulated accessTypeEvent", np.sum(featureMatrix, axis=0)

# nomalizing the column

x = np.array(featureMatrix)
min_max_scaler = MinMaxScaler()
normalizedFeatureMatrix = min_max_scaler.fit_transform(x)


#Plotting
objects = accessTypeList
y_pos = np.arange(numberOfAccessType)
accumulatedAccesses = np.sum(featureMatrix, axis=0)

plt.barh(y_pos, accumulatedAccesses, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accumulated AccessType Events')
plt.title('Summarization of AccessType Events')
plt.show()


# optimal number of cluster finding steps
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
    print "Number of users in this category:", len(userList)
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
# Not effective because of correlated feature
print "#####Gaussian####", outliers
mu, sigma = estimateGaussian(x)
print mu, sigma
p = multivariateGaussian(x,mu,sigma)
ep = 9.036201327981216e-05
outliers = np.asarray(np.where(p < ep))
'''

#Oneclass SVM
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf.fit(x)
pred = clf.predict(x)
# inliers are labeled 1, outliers are labeled -1
unusual = 0
svm_meansList = []
for iter in xrange(0, len(x)):
    if pred[iter]< 0:
        unusual = unusual + 1
        print index_to_userid[iter], featureMatrix[iter]
        svm_meansList.append(index_to_userid[iter])
print "Unusual user according to OneClassSVC:",unusual



#Isolation Forest Algorithm
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


overLapped = 0
overLappedList =  []
for iter in xrange(0, len(x)):
    if y_pred[iter]< 0:
        print index_to_userid[iter], featureMatrix[iter]
        if index_to_userid[iter] in svm_meansList:
            overLapped = overLapped + 1
            overLappedList.append(index_to_userid[iter])

print "OverLapped Anomalous Users from One-Class SVM and Isolation Forest:", len(overLappedList)
index = 0
for user in overLappedList:
    print index, user
    index = index + 1

